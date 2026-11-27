"""Prepare a cloud package from all project-curated senator Markdown sections."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.releases import checksum, prepare_markdown_release
from ai.vector_store import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=Path, default=PROJECT_ROOT / 'data/corpus/processed/senator_chunks.jsonl')
    parser.add_argument('--source-root', type=Path, default=PROJECT_ROOT / 'data/senator_records')
    parser.add_argument('--release-id', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--embedding-provider', choices=('openai', 'fireworks', 'local_hash'), default='openai',
                        help='Embedding identity for both publication and runtime; default: openai.')
    parser.add_argument('--subject', action='append', dest='subjects',
                        help='Optional senator ID filter; repeat to select more than one. Default: every Markdown subject.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError('Use a new, empty output directory; release packages are immutable.')
        chunks, manifest = prepare_markdown_release(
            read_jsonl([args.chunks]), args.release_id, args.collection, args.source_root, args.subjects,
            embedding_provider=args.embedding_provider)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        chunk_output = args.output_dir / 'chunks.jsonl'
        manifest_output = args.output_dir / 'manifest.json'
        if chunk_output.exists() or manifest_output.exists():
            raise ValueError('Release output files already exist; use a new directory.')
        chunk_output.write_text(
            ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in chunks),
            encoding='utf-8')
        manifest['chunks_sha256'] = checksum(chunk_output)
        manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Markdown release preparation failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'prepared_project_curated_markdown', 'rows': len(chunks),
                      'subjects': manifest['expected_counts'], 'release_id': args.release_id,
                      'collection': args.collection}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
