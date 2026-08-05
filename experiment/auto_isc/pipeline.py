#!/usr/bin/env python3
"""
Auto-ISC Pipeline.

Generates harmful datasets by running TVD templates against frontier models.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError
from tqdm.asyncio import tqdm_asyncio

load_dotenv(Path(__file__).parent.parent.parent / ".env")

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
RESULTS_DIR = Path(__file__).parent / "results"

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

async def generate_completion(
    client: AsyncOpenAI, 
    sem: asyncio.Semaphore,
    model: str, 
    template_name: str, 
    prompt_text: str,
    max_retries: int = 3
) -> dict:
    """Generate completion for a single template and model."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.7,
                    max_tokens=2048,
                )
                
                content = response.choices[0].message.content
                
                return {
                    "template": template_name,
                    "model": model,
                    "prompt": prompt_text,
                    "completion": content,
                    "error": None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "template": template_name,
                        "model": model,
                        "prompt": prompt_text,
                        "completion": None,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                await asyncio.sleep(1)

def get_templates(domains: list[str] = None, template_names: list[str] = None) -> list[Path]:
    """Discover templates matching the filter criteria."""
    all_templates = [d for d in TEMPLATES_DIR.iterdir() if d.is_dir()]
    
    selected = []
    for t in all_templates:
        if template_names and t.name not in template_names:
            continue
        if domains:
            domain = t.name.split("_")[0]
            if domain not in domains:
                continue
        selected.append(t)
        
    return sorted(selected)

def load_prompt(template_dir: Path) -> str:
    """Load the best available prompt from the template directory."""
    # Prefer v3 zeroshot, fallback to prompt.txt
    v3_path = template_dir / "prompt_v3_zeroshot.txt"
    if v3_path.exists():
        return v3_path.read_text(encoding="utf-8")
        
    prompt_path = template_dir / "prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
        
    raise FileNotFoundError(f"No prompt found in {template_dir}")

async def main():
    parser = argparse.ArgumentParser(description="Auto-ISC Pipeline Generation")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of OpenRouter models")
    parser.add_argument("--domains", type=str, help="Comma-separated list of domains (e.g., compbio,cyber)")
    parser.add_argument("--templates", type=str, help="Comma-separated list of specific template names")
    parser.add_argument("--output", type=str, default="auto_isc_output.jsonl", help="Output filename")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent API requests")
    
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
        
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(args.concurrent)
    
    models = [m.strip() for m in args.models.split(",")]
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None
    template_names = [t.strip() for t in args.templates.split(",")] if args.templates else None
    
    templates = get_templates(domains, template_names)
    if not templates:
        print("No templates found matching criteria.")
        sys.exit(1)
        
    print(f"Found {len(templates)} templates. Running against {len(models)} models.")
    
    tasks = []
    for t_dir in templates:
        try:
            prompt_text = load_prompt(t_dir)
            for model in models:
                tasks.append(generate_completion(client, sem, model, t_dir.name, prompt_text))
        except Exception as e:
            print(f"Warning: Skipping {t_dir.name} - {e}")
            
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = RESULTS_DIR / args.output
    
    print(f"Executing {len(tasks)} tasks...")
    results = await tqdm_asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r["error"] is None)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(results)} results ({success_count} successful) to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
