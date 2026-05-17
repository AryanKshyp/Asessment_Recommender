"""
convert_traces.py
-----------------
Convert sample conversation markdown (traces_old/*.md) into JSON traces
for test_conversations.py.

Markdown layout:
  ### Turn N
  **User**
  > line one
  > line two
  **Agent**
  ... prose and optional recommendation table ...
  _`end_of_conversation`: **true**_
"""

import json
import re
from pathlib import Path

TURN_SPLIT = re.compile(r"(?=### Turn \d+)", re.IGNORECASE)
USER_BLOCK = re.compile(r"\*\*User\*\*\s*\n+(.*?)(?=\n\*\*Agent\*\*)", re.DOTALL | re.IGNORECASE)
AGENT_BLOCK = re.compile(r"\*\*Agent\*\*\s*\n+(.*)", re.DOTALL | re.IGNORECASE)
EOC_TRUE = re.compile(r"end_of_conversation[`\s:*]*\*\*true\*\*", re.IGNORECASE)
TABLE_ROW = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|")


def extract_user_message(turn_block: str) -> str:
    """Join blockquoted lines under **User** into one message string."""
    match = USER_BLOCK.search(turn_block)
    if not match:
        return ""

    lines: list[str] = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            lines.append(line[1:].lstrip())
        elif lines:
            # Rare continuation line without blockquote prefix
            lines[-1] = f"{lines[-1]} {line}".strip()

    return "\n".join(lines).strip()


def extract_table_names(agent_section: str) -> list[str]:
    """Parse assessment names from markdown recommendation tables."""
    names: list[str] = []
    for line in agent_section.splitlines():
        row = TABLE_ROW.match(line.strip())
        if not row:
            continue
        name = row.group(1).strip()
        if name.lower() == "name":
            continue
        names.append(name)
    return names


def extract_final_assessments(md_content: str) -> list[str]:
    """
    Expected assessments = names from the recommendation table on the
    final turn where end_of_conversation is true.
    """
    turns = [t for t in TURN_SPLIT.split(md_content) if t.strip().startswith("### Turn")]

    for turn in reversed(turns):
        if not EOC_TRUE.search(turn):
            continue
        agent_match = AGENT_BLOCK.search(turn)
        if not agent_match:
            continue
        names = extract_table_names(agent_match.group(1))
        if names:
            return names[:10]

    return []


def parse_md_trace(md_content: str, trace_id: str) -> dict:
    turns = [t for t in TURN_SPLIT.split(md_content) if t.strip().startswith("### Turn")]

    user_messages = [msg for t in turns if (msg := extract_user_message(t))]
    expected_assessments = extract_final_assessments(md_content)

    return {
        "trace_id": trace_id,
        "persona": "Extracted Persona",
        "facts": {},
        "user_messages": user_messages,
        "expected_assessments": expected_assessments,
    }


def convert_all(
    target_dir: Path = Path("./traces_old"),
    output_dir: Path = Path("./traces"),
) -> None:
    output_dir.mkdir(exist_ok=True)

    md_files = sorted(target_dir.glob("C*.md"))
    if not md_files:
        print(f"No C*.md files found in '{target_dir}'.")
        return

    for file_path in md_files:
        trace_id = file_path.stem
        content = file_path.read_text(encoding="utf-8")
        parsed = parse_md_trace(content, trace_id)

        out_path = output_dir / f"{trace_id}.json"
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(
            f"  {trace_id}: {len(parsed['user_messages'])} turns, "
            f"{len(parsed['expected_assessments'])} expected"
        )

    print(f"\nConverted {len(md_files)} traces -> '{output_dir}/'")


if __name__ == "__main__":
    convert_all()
