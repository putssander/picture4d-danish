#!/usr/bin/env python3
"""
USAS Metaphor Analysis Script using aisuite

Analyzes text for conceptual metaphors mapping a specified tenor domain to vehicles,
using USAS semantic tags and an LLM.
"""

import json
import re
import requests
import aisuite as ai
from typing import Dict, List, Any, Optional
import argparse
import sys
from pathlib import Path


def download_usas_tags() -> bool:
    """Download USAS tags from the official source and save as JSON."""
    url = "https://ucrel.lancs.ac.uk/usas/semtags.txt"
    semtags_data: Dict[str, str] = {}
    try:
        response = requests.get(url)
        response.raise_for_status()
        for line in response.text.strip().split("\n"):
            if '\t' in line:
                code, desc = line.split('\t', 1)
                semtags_data[code.strip()] = desc.strip()
        with open("usas_tags.json", 'w', encoding='utf-8') as f:
            json.dump(semtags_data, f, indent=2, ensure_ascii=False)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading USAS tags: {e}")
        return False
    except IOError as e:
        print(f"Error writing USAS tags file: {e}")
        return False


def load_usas_tags() -> Dict[str, Any]:
    """Load USAS tags from JSON, downloading if missing."""
    path = Path("usas_tags.json")
    if not path.exists() and not download_usas_tags():
        print("Failed to obtain USAS tags.")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading USAS tags: {e}")
        sys.exit(1)

# Schema definition (referenced in build_analysis_prompt)
SCHEMA = '''
{
  "metaphor_present": boolean,
  "metaphors": [
    {
      "metaphor_span": string,
      "tenor": {
        "concept": string|null,
        "usas_tag": {"code": string, "description": string},
        "surface": [string, ...],
        "explicit": boolean
      },
      "vehicle": {
        "surface": [string, ...],
        "concept": string|null,
        "basic_usas": {"code": string, "description": string},
        "contextual_usas": {"code": string, "description": string},
        "meaning_discrepancy": string
      },
      "context": string
    }
  ]
}
'''


def build_analysis_prompt(text: str, target_tenor_tag: Optional[str] = None) -> str:
    """
    Build the NL prompt (JSON-only output) referencing the external USAS tags.
    Assumes tags are sent separately in a metadata message named 'tags'.
    """
    focus_line = f"Focusing EXCLUSIVELY on tenor tag '{target_tenor_tag}'." if target_tenor_tag else ""
    missing_tag_check = (
        f"If '{target_tenor_tag}' not in tags, return {json.dumps({'metaphor_present': False, 'metaphors': []})}."
        if target_tenor_tag else ""
    )

    # Add explicit filtering instructions for B2
    strict_filtering = ""
    if target_tenor_tag == "B2":
        strict_filtering = """

### CRITICAL B2 FILTERING RULES:
- ONLY detect metaphors where the TENOR (what is being described) is EXPLICITLY about:
  * Health conditions (cancer, illness, disease, symptoms)
  * Medical treatments (chemotherapy, surgery, medication)
  * Bodily functions or anatomy
  * Medical experiences or processes
- DO NOT detect these as B2 metaphors:
  * "Learning is a journey" → P1 (Education), NOT B2
  * "Time is money" → I1 (Money), NOT B2
  * "Knowledge is power" → P1 (Education), NOT B2
  * "Love is a flame" → E2 (Emotional), NOT B2
  * "Life is a game" → K5 (Games), NOT B2
  * "Politics is a chess match" → K5 (Games), NOT B2
- If the text contains metaphors but NONE have health-related tenors, return metaphor_present: false"""

    examples_data = [
        {
            "metaphor_span": "Like a scary fairground ride – it might be scary in places, but it will eventually stop and you can get off.",
            "tenor": {
                "concept": "cancer experience",
                "usas_tag": {"code": "B2", "description": "Health and disease"},
                "surface": [],
                "explicit": False
            },
            "vehicle": {
                "surface": ["scary fairground ride"],
                "concept": "the frightening yet finite experience of cancer treatment",
                "basic_usas": {"code": "K1", "description": "Entertainment generally"},
                "contextual_usas": {"code": "E5", "description": "Fear/bravery/shock"},
                "meaning_discrepancy": "K1 Entertainment generally vs. E5 Fear/bravery/shock in a medical ordeal"
            },
            "context": "Like a scary fairground ride – it might be scary in places, but it will eventually stop and you can get off."
        },
        {
            "metaphor_span": "My journey with cancer may not be smooth but it certainly makes me look up and take notice of the scenery!",
            "tenor": {
                "concept": "cancer",
                "usas_tag": {"code": "B2", "description": "Health and disease"},
                "surface": ["cancer"],
                "explicit": True
            },
            "vehicle": {
                "surface": ["journey"],
                "concept": "living through cancer with ups and downs and moments of reflection",
                "basic_usas": {"code": "M1", "description": "Moving, coming and going"},
                "contextual_usas": {"code": "A12", "description": "Easy/difficult"},
                "meaning_discrepancy": "M1 Moving, coming and going vs. A12 Easy/difficult of coping with cancer"
            },
            "context": "My journey with cancer may not be smooth but it certainly makes me look up and take notice of the scenery!"
        }
    ]

    return f"""
You are a Metaphor Analysis Assistant.

### Definition
A metaphor maps two domains: TENOR → VEHICLE.

### TENOR vs VEHICLE (Critical Distinction):
- **TENOR**: The abstract, complex, or difficult-to-explain concept being described
  * Often intangible, emotional, or experiential
  * Examples: illness experience, love, learning process, economic situation
  * Usually the target domain we want to understand better
  
- **VEHICLE**: The concrete, familiar, easily understood concept used to explain the tenor
  * Drawn from everyday physical experience
  * Examples: journey, battle, building, game, fire
  * Provides structure and understanding for the abstract tenor

### Metaphor Examples:
- "Cancer is a journey" → TENOR: cancer experience (abstract, complex) → VEHICLE: journey (concrete, familiar)
- "Love is a flame" → TENOR: love (abstract emotion) → VEHICLE: flame (concrete, physical)
- "Learning is building knowledge" → TENOR: learning process (abstract) → VEHICLE: building (concrete activity)

### Instructions
0. Validate output against the schema. If invalid, return an error object.
1. {missing_tag_check}
2. {focus_line}
3. STRICT FILTERING: Only detect metaphors where the tenor's USAS tag EXACTLY equals "{target_tenor_tag}".
4. If the text contains metaphors but NONE match the target tenor tag, return metaphor_present: false.
5. Double-check: Does the tenor (the abstract concept being described) truly belong to tag "{target_tenor_tag}"?

### Input
{{
  "text": {json.dumps(text)},
  "target_tenor_tag": {json.dumps(target_tenor_tag)}
}}

### Schema
{SCHEMA}

### Rules
- JSON only: no extra whitespace, comments, or trailing commas.
- Include only schema fields.
- MANDATORY: tenor.usas_tag.code must exactly match target_tenor_tag
- MANDATORY: context field must be at least 1-2 complete sentences with adequate context
- REMEMBER: Tenor = abstract/complex concept, Vehicle = concrete/familiar concept
- If no metaphors match target tag, return {{"metaphor_present": false, "metaphors": []}}

### Examples (will improve adherence)
{json.dumps(examples_data, indent=2)}
""".strip()


def extract_json(response: str) -> Any:
    """Extract JSON from LLM response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if m:
        response = m.group(1)
    try:
        return json.loads(response)
    except Exception:
        print("Invalid JSON from LLM.")
        return {"error": "invalid JSON"}


def analyze_text(
    client: ai.Client,
    model: str,
    text: str,
    target: Optional[str] = None
) -> Dict[str, Any]:
    prompt = build_analysis_prompt(text, target)
    messages = [
        {"role": "system", "content": "You are a Metaphor Analysis Assistant."},
        {"role": "assistant", "name": "tags", "content": json.dumps(load_usas_tags())},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_k=1,
        top_p=1.0,
    )
    return extract_json(resp.choices[0].message.content)


def batch_analyze(
    client: ai.Client,
    model: str,
    texts: List[str],
    target: Optional[str]
) -> List[Any]:
    return [analyze_text(client, model, t, target) for t in texts]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze text for metaphors with USAS tags."
    )
    parser.add_argument("--text", help="Single text to analyze")
    parser.add_argument(
        "--texts-file",
        help="File with one text per line"
    )
    parser.add_argument("--model", default="ollama:gemma3:27b")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--target-tenor", help="USAS tag code for tenor filter")
    args = parser.parse_args()

    if not args.text and not args.texts_file:
        print("Provide --text or --texts-file.")
        sys.exit(1)

    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    if not texts:
        print("No texts to analyze.")
        sys.exit(1)

    client = ai.Client()
    client.configure({"ollama": {"timeout": 600}})

    if len(texts) == 1:
        results = analyze_text(client, args.model, texts[0], args.target_tenor)
    else:
        results = batch_analyze(client, args.model, texts, args.target_tenor)

    output = {"model": args.model, "results": results}
    data = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"Results saved to {args.output}")
    else:
        print(data)


if __name__ == "__main__":
    main()
