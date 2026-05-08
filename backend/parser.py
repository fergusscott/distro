import os
import json
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an insurance submission parser for an MGA.
Extract structured risk data from the submission text provided.
Return ONLY valid JSON with these exact keys:

{
  "insured_name": string,
  "class_of_business": one of [general_liability, commercial_property, professional_liability, workers_comp, commercial_auto, cyber, management_liability],
  "industry": one of [construction, manufacturing, retail, technology, healthcare, hospitality, real_estate, financial_services, transportation],
  "state": two-letter US state code,
  "premium": number (estimated annual premium in USD, infer if not stated),
  "loss_ratio": number between 0 and 1 (prior loss ratio if available, else estimate 0.4),
  "years_in_business": integer,
  "prior_claims": integer (number of claims in last 3 years, 0 if not mentioned),
  "num_employees": integer,
  "coverage_limit": number (primary limit in USD),
  "notes": string (any relevant details not captured above)
}

If a field cannot be determined from the text, use a sensible default. Never omit a field."""


def _clean(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def parse_submission(text: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return _clean(message.content[0].text.strip())


def parse_pdf(pdf_bytes: bytes) -> dict:
    import base64
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract the insurance submission data from this ACORD form.",
                },
            ],
        }],
    )
    return _clean(message.content[0].text.strip())
