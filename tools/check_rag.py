"""Read-only release health check. No embedding or generation calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.rag_config import RAGConfig, RAGConfigurationError
from ai.rag_factory import build_retriever
from ai.releases import validate_package, verify_collection, checksum, require
from ai.vector_store import read_jsonl
from tools.runtime_settings import load_settings, read_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-manifest', type=Path)
    parser.add_argument('--chunks', type=Path)
    parser.add_argument('--secrets-file', type=Path)
    parser.add_argument('--runtime-settings', action='store_true',
                        help='Validate a private TOML export with the same rules and read-only key name as Streamlit Secrets.')
    parser.add_argument('--config-only', action='store_true', help='Validate settings offline; make no database or model requests.')
    parser.add_argument('--connection-only', action='store_true', help='Check Qdrant authentication without a published release or model credentials.')
    args = parser.parse_args()
    try:
        if args.runtime_settings:
            require(args.secrets_file, '--runtime-settings requires an explicit private TOML file.')
            config = RAGConfig.load(settings=read_settings(args.secrets_file), require_cloud=True,
                                    require_credentials=args.config_only)
        else:
            load_settings(args.secrets_file)
            config = RAGConfig.load(require_credentials=args.config_only)
        require(config.backend != 'local', 'Health check requires cloud mode.')
        if args.config_only:
            print(json.dumps({'status': 'configuration_ok', 'model_calls': 0, 'database_calls': 0,
                              'note': 'Settings accepted; credentials, collection and model access are not authenticated.'}))
            return 0
        if args.connection_only:
            require(config.backend == 'qdrant_cloud' and bool(config.qdrant_api_key), 'Qdrant credentials are required.')
            client = config.cloud_client()
            try:
                client.get_collections()
            finally:
                client.close()
            print(json.dumps({'status': 'connection_ok', 'model_calls': 0, 'writes': 0,
                              'note': 'Authentication succeeded; this does not verify a populated release or key write restrictions.'}))
            return 0
        require(args.release_manifest and args.chunks, 'Supply a release manifest and chunks, or --connection-only.')
        manifest = json.loads(args.release_manifest.read_text(encoding='utf-8'))
        require(checksum(args.chunks) == manifest.get('chunks_sha256'), 'Chunk checksum mismatch.')
        rows = validate_package(read_jsonl([args.chunks]), manifest, config)
        store = build_retriever(config)._get_store()
        if config.backend == 'qdrant_cloud':
            from ai.qdrant_store import verify_qdrant_collection
            verify_qdrant_collection(store.client, config.collection, rows)
        else:
            verify_collection(store.collection, rows)
    except RAGConfigurationError as exc:
        print(f'Configuration check failed: {exc}', file=sys.stderr)
        return 2
    except Exception:
        print('Release health check failed. Check settings, connection and release identity privately; no model calls were made.', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'healthy', 'release_id': config.release, 'rows': len(rows), 'model_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
