"""Build the clean, joined email-analysis table used by the project.

The raw corpus folders verify the CSV labels and provide source metadata. They
are joined to the flattened CSV with Message-ID. Folder labels are deliberately
excluded from model features because they would reveal the answer.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime, parseaddr
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT / "data" / "raw" / "archive"
CSV_PATH = RAW_ROOT / "Spamdataset.csv"
MAIL_ROOT = RAW_ROOT / "spam_data" / "spam_data"
OUTPUT_DIR = PROJECT / "data" / "processed"

LABEL_GROUPS = {"easy_ham": 0, "easy_ham_2": 0, "spam": 1, "spam_2": 1}
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def read_message(raw: bytes):
    """Parse an mbox-style message while removing the envelope line if present."""
    if raw.startswith(b"From ") and b"\n" in raw:
        raw = raw.split(b"\n", 1)[1]
    return BytesParser(policy=policy.compat32).parsebytes(raw)


def normalized_message_id(message) -> str:
    return (message.get("Message-ID") or "").strip().lower()


def decode_part(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        payload = part.get_payload()
        return payload if isinstance(payload, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def extract_body(message) -> tuple[str, bool, int]:
    """Return readable body text, whether HTML exists, and attachment count."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachment_count = 0

    for part in message.walk():
        if part.is_multipart():
            continue
        content_type = (part.get_content_type() or "").lower()
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        if disposition == "attachment" or filename:
            attachment_count += 1
            continue
        text = decode_part(part)
        if content_type == "text/plain":
            plain_parts.append(text)
        elif content_type == "text/html":
            html_parts.append(text)

    has_html = bool(html_parts)
    if plain_parts:
        body = "\n".join(plain_parts)
    else:
        body = "\n".join(TAG_RE.sub(" ", html.unescape(part)) for part in html_parts)
    return SPACE_RE.sub(" ", body).strip(), has_html, attachment_count


def safe_datetime(value: str | None):
    if not value:
        return pd.NaT
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return pd.NaT


def top_level_domain(address: str) -> str:
    _, parsed = parseaddr(address or "")
    if "@" not in parsed:
        return "missing"
    domain = parsed.rsplit("@", 1)[1].lower().strip(".>")
    if "." not in domain:
        return "other"
    return "." + domain.rsplit(".", 1)[1]


def count_recipients(message) -> int:
    values = []
    for field in ("To", "Cc", "Bcc"):
        values.extend(message.get_all(field, []))
    return len([address for _, address in getaddresses(values) if address])


def text_rates(text: str) -> dict[str, float | int]:
    characters = max(len(text), 1)
    letters = [char for char in text if char.isalpha()]
    letter_count = max(len(letters), 1)
    return {
        "text_char_count": len(text),
        "text_word_count": len(re.findall(r"\b\w+\b", text)),
        "url_count": len(URL_RE.findall(text)),
        "uppercase_rate": sum(char.isupper() for char in letters) / letter_count,
        "exclamation_rate": text.count("!") / characters,
        "dollar_rate": text.count("$") / characters,
        "digit_rate": sum(char.isdigit() for char in text) / characters,
    }


def parse_csv_row(row: pd.Series) -> dict:
    raw = row["emails"].encode("utf-8", errors="replace")
    message = read_message(raw)
    subject = str(message.get("Subject") or "").strip()
    body, has_html, attachment_count = extract_body(message)
    date_value = safe_datetime(message.get("Date"))
    if pd.isna(date_value):
        sent_hour, sent_weekday, timezone_offset = np.nan, "missing", np.nan
    else:
        sent_hour = date_value.hour
        sent_weekday = date_value.strftime("%A")
        offset = date_value.utcoffset()
        timezone_offset = offset.total_seconds() / 3600 if offset else 0

    text = f"{subject} {body}".strip()
    features = text_rates(text)
    message_id = normalized_message_id(message)
    # Message-ID is unique in the merged corpus. Hash it so the public export
    # can identify a point without exposing the original identifier.
    record_hash = hashlib.sha256(message_id.encode("utf-8", errors="replace")).hexdigest()[:16]
    return {
        "message_id": message_id,
        "record_id": f"email_{record_hash}",
        "label": int(row["label"]),
        "subject": subject,
        "body_text": body,
        "model_text": text,
        "sent_hour": sent_hour,
        "sent_weekday": sent_weekday,
        "timezone_offset": timezone_offset,
        "sender_tld": top_level_domain(message.get("From")),
        "content_type": (message.get_content_type() or "missing").lower(),
        "is_html": int(has_html),
        "has_attachment": int(attachment_count > 0),
        "attachment_count": attachment_count,
        "recipient_count": count_recipients(message),
        "received_hops": len(message.get_all("Received", [])),
        "has_reply_to": int(bool(message.get("Reply-To"))),
        "has_list_unsubscribe": int(bool(message.get("List-Unsubscribe"))),
        "is_multipart": int(message.is_multipart()),
        "subject_char_count": len(subject),
        **features,
    }


def build_raw_index() -> pd.DataFrame:
    rows = []
    for group, label in LABEL_GROUPS.items():
        folder = MAIL_ROOT / group
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            message = read_message(raw)
            message_id = normalized_message_id(message)
            rows.append(
                {
                    "message_id": message_id,
                    "raw_group": group,
                    "raw_label": label,
                    "raw_filename": path.name,
                }
            )
    index = pd.DataFrame(rows)
    return index[index["message_id"].ne("")].drop_duplicates("message_id")


def main() -> None:
    if not CSV_PATH.exists() or not MAIL_ROOT.exists():
        raise FileNotFoundError(
            "Expected raw archive contents under data/raw/archive/. See README.md."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_csv = pd.read_csv(CSV_PATH)
    if set(source_csv.columns) != {"emails", "label"}:
        raise ValueError(f"Unexpected CSV columns: {list(source_csv.columns)}")

    flattened = pd.DataFrame([parse_csv_row(row) for _, row in source_csv.iterrows()])
    raw_index = build_raw_index()
    merged = flattened.merge(raw_index, on="message_id", how="inner", validate="one_to_one")

    if not (merged["label"] == merged["raw_label"]).all():
        raise ValueError("Raw-folder labels and CSV labels disagree; stop before modeling.")
    if merged["record_id"].duplicated().any():
        raise ValueError("Duplicate anonymous record IDs detected after join.")

    # Source labels confirm the merge but are never model features or web data.
    model_columns = [column for column in merged.columns if column not in {"raw_group", "raw_label", "raw_filename"}]
    model_data = merged[model_columns].copy()
    model_data.to_csv(OUTPUT_DIR / "emails_merged_full.csv", index=False)

    safe_features = model_data.drop(columns=["subject", "body_text", "model_text", "message_id"])
    safe_features.to_csv(OUTPUT_DIR / "emails_merged_features.csv", index=False)

    quality = {
        "csv_records": int(len(source_csv)),
        "raw_messages_with_message_id": int(len(raw_index)),
        "joined_records": int(len(model_data)),
        "excluded_csv_records": int(len(source_csv) - len(model_data)),
        "label_counts": {str(key): int(value) for key, value in model_data["label"].value_counts().sort_index().items()},
        "missingness": {key: int(value) for key, value in model_data.isna().sum().items()},
        "feature_groups": {
            "text_style": [
                "subject_char_count", "text_char_count", "text_word_count", "url_count",
                "uppercase_rate", "exclamation_rate", "dollar_rate", "digit_rate",
            ],
            "context_numeric": [
                "sent_hour", "timezone_offset", "attachment_count", "recipient_count", "received_hops",
            ],
            "context_binary": ["is_html", "has_attachment", "has_reply_to", "has_list_unsubscribe", "is_multipart"],
            "context_categorical": ["sent_weekday", "sender_tld", "content_type"],
        },
    }
    (OUTPUT_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"Built {len(model_data):,} joined records.")
    print("Verified labels: 0 = legitimate/ham, 1 = spam.")
    print(f"Saved full and public-safe local tables to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
