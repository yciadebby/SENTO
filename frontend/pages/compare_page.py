from html import escape
import re

import streamlit as st

from ai.validator import is_safe_source_url
from backend.comparison_service import COMPARISON_CATEGORIES, PERIODS
from frontend.api import get_comparison_records
from frontend.components.senator_card import avatar


def _format_inline_markdown(text: str) -> str:
    """Render escaped inline Markdown (links, bold, italic, code) safely."""
    placeholders: list[str] = []

    def _stash(html_snippet: str) -> str:
        token = f"\x00MD{len(placeholders)}\x00"
        placeholders.append(html_snippet)
        return token

    def _replace_link(match: re.Match) -> str:
        label = escape(match.group(1))
        url = match.group(2).strip()
        if is_safe_source_url(url):
            return _stash(
                f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label} ↗</a>'
            )
        return label

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", _replace_link, text)
    escaped = escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    for index, snippet in enumerate(placeholders):
        escaped = escaped.replace(f"\x00MD{index}\x00", snippet)
    return escaped


def _render_markdown_html(text: str) -> str:
    """Convert claim Markdown text into safe HTML supporting headings, lists, tables, and links."""
    lines = str(text or "").splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Markdown table
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1])
        ):
            headers = [cell.strip() for cell in stripped.strip("|").split("|")]
            i += 2
            rows_html = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                rows_html.append(
                    "<tr>" + "".join(f"<td>{_format_inline_markdown(cell)}</td>" for cell in cells) + "</tr>"
                )
                i += 1
            head_html = "<tr>" + "".join(f"<th>{_format_inline_markdown(h)}</th>" for h in headers) + "</tr>"
            blocks.append(f'<table class="comparison-md-table"><thead>{head_html}</thead><tbody>{"".join(rows_html)}</tbody></table>')
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = min(6, max(4, len(heading_match.group(1)) + 2))
            blocks.append(f"<h{level} class=\"comparison-md-heading\">{_format_inline_markdown(heading_match.group(2))}</h{level}>")
            i += 1
            continue

        # Bulleted list
        if re.match(r"^[-*+]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                item_text = re.sub(r"^\s*[-*+]\s+", "", lines[i]).strip()
                items.append(f"<li>{_format_inline_markdown(item_text)}</li>")
                i += 1
            blocks.append(f'<ul class="comparison-md-list">{"".join(items)}</ul>')
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item_text = re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()
                items.append(f"<li>{_format_inline_markdown(item_text)}</li>")
                i += 1
            blocks.append(f'<ol class="comparison-md-list">{"".join(items)}</ol>')
            continue

        # Paragraph
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#") or nxt.startswith("|") or re.match(r"^([-*+]|\d+\.)\s+", nxt):
                break
            para_lines.append(nxt)
            i += 1
        blocks.append(f"<p>{_format_inline_markdown(' '.join(para_lines))}</p>")

    return "".join(blocks) or "<p>Claim text unavailable.</p>"


def _status_badge(claim: dict) -> str:
    raw_status = str(claim.get("evidence_status") or "UNSPECIFIED").strip().upper()
    verification_status = str(claim.get("verification_status") or "").strip().lower()
    if raw_status == "VERIFIED" and verification_status == "verified":
        return '<span class="comparison-badge comparison-badge-verified">VERIFIED</span>'
    if raw_status == "VERIFIED":
        return '<span class="comparison-badge comparison-badge-verified-candidate">VERIFIED · CANDIDATE SOURCE</span>'
    if raw_status in {"PARTIALLY_VERIFIED", "DOCUMENTED", "OFFICIAL_STATEMENT"}:
        label = raw_status.replace("_", " ")
        return f'<span class="comparison-badge comparison-badge-partial">{escape(label)}</span>'
    if raw_status in {"ALLEGED", "CONTESTED", "DISPUTED", "PENDING_INVESTIGATION"}:
        label = raw_status.replace("_", " ")
        return f'<span class="comparison-badge comparison-badge-contested">{escape(label)}</span>'
    if raw_status == "UNSPECIFIED":
        return '<span class="comparison-badge comparison-badge-neutral">CLASSIFICATION NOT SPECIFIED</span>'
    label = raw_status.replace("_", " ")
    return f'<span class="comparison-badge comparison-badge-neutral">{escape(label)}</span>'


def _source_link(source, index):
    title = escape(str(source.get("title") or "Source title unavailable"))
    issuer = escape(str(source.get("issuing_body") or "Issuing body unavailable"))
    publication_date = escape(str(source.get("publication_date") or source.get("publication_date_text") or "publication date unavailable"))
    retrieval_date = escape(str(source.get("retrieved_at") or "retrieval date unavailable"))
    locator = escape(str(source.get("locator") or "locator unavailable"))
    verification = "verified provenance" if source.get("verification_status") == "verified" else "candidate source"
    source_type = {
        "government_publication": "Government publication",
        "secondary_news": "News report",
        "research_organization": "Research organization",
        "other": "Other source",
    }.get(source.get("source_type"), "Source type unavailable")
    source_type = escape(source_type)
    url = source.get("source_url")
    label = f"Source {index}: {title}"
    link = (
        f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label} ↗</a>'
        if is_safe_source_url(url)
        else f'<span>{label} · no validated source link</span>'
    )
    return (
        '<div class="comparison-source">'
        f'{link}<small>{escape(verification)} · {source_type} · {issuer} · published {publication_date} · '
        f'retrieved {retrieval_date} · {locator}</small>'
        '</div>'
    )


def _claim_html(claim):
    statement_html = _render_markdown_html(claim.get("text") or "Claim text unavailable.")
    period = escape(str(claim.get("period") or "Period not stated"))
    detail_items = claim.get("details", [])
    detail_html = "".join(
        f'<p class="comparison-detail"><strong>{escape(str(item["label"]))}:</strong> '
        f'{_format_inline_markdown(str(item["value"]))}</p>'
        for item in detail_items
    )
    if len(detail_items) > 2:
        detail_block = (
            f'<details class="comparison-details-box"><summary>Record details ({len(detail_items)})</summary>'
            f'{detail_html}</details>'
        )
    else:
        detail_block = detail_html
    sources = "".join(
        _source_link(source, index)
        for index, source in enumerate(claim.get("sources", []), start=1)
    )
    return (
        '<article class="comparison-claim">'
        f'<div class="comparison-claim-body">{statement_html}</div>{detail_block}'
        f'<div class="comparison-claim-meta"><span>Period: {period}</span>'
        f'{_status_badge(claim)}</div>'
        f'{sources}</article>'
    )


def _category_cell(record, category_key):
    section = record["sections"][category_key]
    name = escape(str(record["senator"]["name"]))
    claims = section["claims"]
    count = len(claims)
    verified_count = section.get("verified_count", 0)
    other_count = count - verified_count
    if count:
        breakdown = f" ({verified_count} verified, {other_count} other status)" if other_count else f" ({verified_count} verified)"
        summary = f'<p class="comparison-coverage">{count} documented {"record" if count == 1 else "records"} in this window{breakdown}.</p>'
    else:
        summary = '<p class="comparison-coverage">0 documented records in this window.</p>'
        summary += '<p class="comparison-gap">No record is present in these files for this category and period. Missing data does not mean no activity occurred.</p>'
    outside = section.get("outside_period_count", section.get("verified_outside_period", 0))
    if outside:
        summary += f'<p class="comparison-gap">{outside} documented {"record is" if outside == 1 else "records are"} in other periods.</p>'
    return (
        f'<div class="comparison-cell"><span class="mobile-person">{name}</span>'
        f'{summary}{"".join(_claim_html(claim) for claim in claims)}</div>'
    )


def _comparison_sources_html(records):
    seen = set()
    cards = []
    index = 0
    for record in records:
        for section in record["sections"].values():
            for claim in section["claims"]:
                for source in claim.get("sources", []):
                    source_id = source.get("source_id")
                    if not source_id or source_id in seen:
                        continue
                    seen.add(source_id)
                    index += 1
                    cards.append(
                        f'<article class="source-detail-card"><h4>{escape(str(record["senator"]["name"]))} · '
                        f'{escape(str(source.get("title") or "Source title unavailable"))}</h4>'
                        f'<p>{escape(str(source.get("issuing_body") or "Issuing body unavailable"))} · '
                        f'{escape(str(source.get("publication_date") or "publication date unavailable"))} · '
                        f'{escape(str(source.get("relevant_period") or claim.get("period") or "Period not stated"))}</p>'
                        f'<p class="source-detail-meta">Locator: {escape(str(source.get("locator") or "not stated"))}</p>'
                        f'{_source_link(source, index)}</article>'
                    )
    if not cards:
        cards.append('<p class="comparison-gap">No source-complete references are available for this comparison.</p>')
    return '<section class="comparison-section comparison-sources" id="compare-sources"><h3>Sources</h3>' + ''.join(cards) + '</section>'


@st.dialog("Sources used in this comparison", width="large")
def _comparison_sources(records):
    st.caption("Sources below support the displayed comparison records. Verification status, issuing body, and locator are shown for each reference.")
    seen = set()
    source_index = 0
    for record in records:
        for section in record["sections"].values():
            for claim in section["claims"]:
                for source in claim.get("sources", []):
                    source_id = source.get("source_id")
                    if not source_id or source_id in seen:
                        continue
                    seen.add(source_id)
                    source_index += 1
                    st.html(
                        f'<article class="source-detail-card"><h2>{escape(str(record["senator"]["name"]))} · '
                        f'{escape(str(source.get("title") or "Source title unavailable"))}</h2>'
                        f'<p>{escape(str(source.get("issuing_body") or "Issuing body unavailable"))} · '
                        f'{escape(str(source.get("publication_date") or "publication date unavailable"))} · '
                        f'{escape(str(source.get("relevant_period") or claim.get("period") or "Period not stated"))}</p>'
                        f'<p class="source-detail-meta">Locator: {escape(str(source.get("locator") or "not stated"))}</p>'
                        f'{_source_link(source, source_index)}</article>'
                    )
    if not seen:
        st.info("No source-linked records are available for the selected senators and period.")


def render(service):
    if st.session_state.get("comparison_count") not in (2, 3):
        st.session_state.comparison_count = 3
    count = st.session_state.get("comparison_count", 3)
    period_key = st.session_state.get("comparison_period", "20th_congress")
    if period_key not in PERIODS:
        period_key = "20th_congress"
        st.session_state.comparison_period = period_key

    with st.container(key="compare_panel"):
        st.html(
            '<div class="compare-heading"><h1>Compare Senators</h1>'
            '<p>Explore the same public-record categories side by side, with source and period details.</p></div>'
        )
        with st.container(key="compare_toolbar"):
            period, links, note = st.columns([284, 672, 300], gap="small")
            with period:
                period_label = "20th Congress · through 25 Nov 2026" if period_key == "20th_congress" else "All periods · through 25 Nov 2026"
                with st.popover(period_label):
                    count = st.radio("Number of senators", [2, 3], horizontal=True, key="comparison_count")
                    st.radio(
                        "Record period",
                        ["20th_congress", "all_periods"],
                        format_func=lambda value: "20th Congress (2025–2026 cutoff)" if value == "20th_congress" else "All researched periods",
                        key="comparison_period",
                    )
                    st.caption("Research files are dated through 25 November 2026. Claims keep their own period labels; where a file gives only a year, the window can only be matched at year precision.")
            with links:
                nav_labels = {
                    "background": "Profile",
                    "legislation": "Bills",
                    "committees": "Committees",
                    "votes": "Votes",
                    "positions": "Positions",
                    "disclosures": "Disclosures",
                    "accountability": "Accountability",
                    "performance": "Performance",
                }
                nav_items = [(key, nav_labels[key]) for key in COMPARISON_CATEGORIES]
                st.html(
                    '<nav class="compare-links" aria-label="Comparison sections">'
                    + "".join(f'<a href="#compare-{key}" data-compare-scroll="true">{escape(label)}</a>' for key, label in nav_items)
                    + '<a href="#compare-sources" data-compare-scroll="true">Sources</a></nav>'
                    + '<script>if(!window.__compareScrollBound){window.__compareScrollBound=true;document.addEventListener("click",function(event){const link=event.target.closest(".compare-links a[data-compare-scroll]");if(!link)return;const target=document.querySelector(link.getAttribute("href"));if(!target)return;event.preventDefault();target.scrollIntoView({behavior:window.matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth",block:"start"});});}</script>',
                    unsafe_allow_javascript=True,
                )
            with note:
                st.html('<p class="compare-note">No rankings or endorsements · gaps stay visible</p>')

        ids = [senator["id"] for senator in service.senators]
        stored_ids = list(st.session_state.get("compare_ids", []))
        chosen = []
        for senator_id in stored_ids:
            if senator_id in ids and senator_id not in chosen:
                chosen.append(senator_id)
            if len(chosen) == count:
                break
        for senator_id in ids:
            if len(chosen) == count:
                break
            if senator_id not in chosen:
                chosen.append(senator_id)

        with st.container(key="compare_selectors"):
            for index, column in enumerate(st.columns(count, gap="large")):
                with column:
                    options = [sid for sid in ids if sid not in chosen or sid == chosen[index]]
                    selected = st.selectbox(
                        f"Senator {index + 1}",
                        options,
                        index=options.index(chosen[index]),
                        format_func=lambda sid: service.get(sid)["name"],
                        key=f"compare_select_{index}",
                        label_visibility="collapsed",
                    )
                    if selected != chosen[index]:
                        new_ids = chosen.copy()
                        new_ids[index] = selected
                        st.session_state.compare_ids = new_ids + [sid for sid in ids if sid not in new_ids][:3 - len(new_ids)]
                        st.rerun()
        st.session_state.compare_ids = chosen + [sid for sid in ids if sid not in chosen][:3 - len(chosen)]

        records = get_comparison_records(service, chosen, period=period_key)
        if st.session_state.pop("show_compare_sources", False):
            _comparison_sources(records)

        people = "".join(
            '<div class="comparison-person">'
            + avatar(record["senator"], "large")
            + f'<h3>{escape(str(record["senator"]["name"]))}</h3>'
            + f'<p>{escape(str(record["period"]))}</p>'
            + f'<p>{record.get("total_claim_count", record["verified_claim_count"])} records ({record["verified_claim_count"]} verified) · {record["source_count"]} sources · '
            + f'research status: {escape(str(record["research_status"]))} · '
            + f'coverage: {escape(str(record["research_depth"]))}</p>'
            + (
                f'<details class="comparison-research-note"><summary>Profile research note</summary>'
                f'<p>{escape(str(record["research_note"]))}</p></details>'
                if record.get("research_note") else ""
            )
            + '</div>'
            for record in records
        )
        rows = []
        for key, category in COMPARISON_CATEGORIES.items():
            cells = "".join(_category_cell(record, key) for record in records)
            rows.append(
                f'<section class="comparison-section" id="compare-{key}">'
                f'<h3>{escape(category["title"])}</h3>'
                f'<div class="comparison-grid">{cells}</div></section>'
            )
        st.html(
            f'<div class="comparison-scroll" role="region" aria-label="Side-by-side senator records" tabindex="0" style="--count:{count}">'
            '<div class="compare-data-note">All documented claims from each senator’s research file are shown, ranked with VERIFIED records first. '
            'Each entry keeps its original classification label; missing data does not mean no record exists.</div>'
            + f'<div class="comparison-grid comparison-portraits">{people}</div>{"".join(rows)}{_comparison_sources_html(records)}</div>'
        )
