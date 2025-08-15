import argparse
import pandas as pd
import asyncio
import json
from asyncio import Semaphore
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional

# Pydantic models for structured output
class ErrorItem(BaseModel):
    """
    Individual error item with position, severity, and category information.
    """
    start: int  # Starting character index in translation (0-based)
    end: int  # Ending character index in translation (0-based, exclusive)
    severity: str  # "Major" or "Minor"
    category: str  # Format: "category/subcategory"
    
class ErrorItem(BaseModel):
    """
    Individual error item with position, severity, and category information.
    """
    start: int  # Starting character index in translation (0-based)
    end: int  # Ending character index in translation (0-based, exclusive)
    severity: str  # "Major" or "Minor"
    category: str  # Format: "category/subcategory"

class ErrorResponse(BaseModel):
    """
    Response model for GEMBA translation quality evaluation.
    
    STRICT JSON FORMAT - NO COMMENTS OR EXPLANATIONS ALLOWED
    
    Structure:
    {
        "errors": [
            {"start": 10, "end": 20, "severity": "Major", "category": "category/subcategory"},
            {"start": 30, "end": 35, "severity": "Minor", "category": "category/subcategory"}
        ]
    }
    
    For omission errors, start and end indicate where the missing content should be inserted.
    """
    errors: Optional[List[ErrorItem]] = []

few_shots = {
    "ende": {
            "source_lang": "English",
            "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
            "target_lang": "German",
            "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
            "answer": '''{
"errors": [
    {"start": 53, "end": 53, "severity": "Major", "category": "accuracy/omission"},
    {"start": 173, "end": 177, "severity": "Minor", "category": "fluency/grammar"},
    {"start": 258, "end": 261, "severity": "Minor", "category": "fluency/register"},
    {"start": 262, "end": 273, "severity": "Major", "category": "accuracy/mistranslation"}
]
}'''
        },
    "encs": {
            "source_lang": "English",
            "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
            "target_lang": "Czech",
            "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
            "answer": '''{
"errors": [
    {"start": 12, "end": 20, "severity": "Major", "category": "accuracy/addition"},
    {"start": 79, "end": 86, "severity": "Minor", "category": "terminology/inappropriate for context"},
    {"start": 149, "end": 149, "severity": "Major", "category": "accuracy/omission"}
]
}'''
        },
    "zhen": {
            "source_lang": "Chinese",
            "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
            "target_lang": "English",
            "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
            "answer": '''{
"errors": [
    {"start": 142, "end": 147, "severity": "Minor", "category": "style/awkward"},
    {"start": 148, "end": 166, "severity": "Major", "category": "accuracy/addition"},
    {"start": 203, "end": 220, "severity": "Major", "category": "accuracy/mistranslation"}
]
}'''
        },
    "enes": {
            "source_lang": "English",
            "source_seg": "According to the terms outlined in the agreement, the supplier shall deliver all components no later than thirty days after receiving the initial purchase order, and any delays must be communicated in writing at least five business days in advance.",
            "target_lang": "Spanish",
            "target_seg": "De acuerdo con los términos establecidos en el acuerdo, el proveedor deberá entregar todos los componentes a más tardar treinta días después de recibir la orden de compra inicial, y cualquier retraso deberá comunicarse por escrito con al menos cinco días hábiles de antelación.",
            "answer": '''{
"errors": []
}'''
        },
}

language_mapping = {
    'en': 'English',
    'en_GB': 'English',
    'en_US': 'English',
    'cs': 'Czech',
    'cs_CZ': 'Czech',
    'ja': 'Japanese',
    'ja_JP': 'Japanese',
    'is_IS': 'Icelandic',
    'de_DE': 'German',
    'uk_UA': 'Ukrainian',
    'uk': 'Ukrainian',
    'ar_EG': 'Arabic',
    'bho_IN': 'Bhojpuri',
    'et_EE': 'Estonian',
    'ko_KR': 'Korean',
    'ru_RU': 'Russian',
    'ru': 'Russian',
    'zh_CN': 'Chinese',
    'zh': 'Chinese',
    'it_IT': 'Italian',
    'sr_Cyrl_RS': 'Serbian',
    'mas_KE': 'Maasai',
    'es': 'Spanish',
    'es_ES': 'Spanish',
    'hi': 'Hindi',
    'hi_IN': 'Hindi'
}


def build_prompt(src_line: str, src_lang: str,
                 tgt_line: str, tgt_lang: str):
    system_prompt = """
You are a careful and balanced annotator for machine translation quality. Your task is to identify translation errors with appropriate confidence.

## EVALUATION GUIDELINES:
- Be thorough but precise: Only mark errors when you are confident they are incorrect
- Consider context and domain: Some variations may be acceptable depending on context
- Distinguish between errors and acceptable alternatives: Multiple valid translations may exist
- Focus on clear, objective errors rather than subjective preferences
- Verify each potential error against the source text before marking
- When in doubt, err on the side of not marking an error

## Error Categories:
- Accuracy: addition, mistranslation, omission, untranslated text
- Fluency: character encoding, grammar, inconsistency, punctuation, register, spelling
- Style: awkward phrasing
- Terminology: inappropriate for context, inconsistent use
- Other: non-translation, other issues

## Severity Classification:
- Major: Errors that impact meaning or usability but do not render the text unusable
- Minor: Errors that do not impact meaning or usability

## CRITICAL OUTPUT REQUIREMENTS:
- Mark errors only when you have clear evidence they are incorrect
- Consider whether alternative translations could be equally valid
- Apply strict standards: better to miss a minor error than create a false positive
- NO comments, explanations, or additional text
- **Mark only the minimal substring that contains the clear error; do not include extra context**
""".strip()

    prompts = [{"role": "system", "content": system_prompt}]

    shot_tpl = "**Source ({source_lang}):** {source_seg}\n**Translation ({target_lang}):** {target_seg}"
    for shot in few_shots.values():
        prompts.append({"role": "user",
                        "content": shot_tpl.format(**shot)})
        prompts.append({"role": "assistant",
                        "content": shot["answer"].strip()})

    query = {
        "source_lang": language_mapping.get(src_lang, src_lang),
        "source_seg": src_line,
        "target_lang": language_mapping.get(tgt_lang, tgt_lang),
        "target_seg": tgt_line
    }
    user_msg = (
        "**Source ({source_lang}):** {source_seg}\n"
        "**Translation ({target_lang}):** {target_seg}"
    ).format(**query)

    prompts.append({"role": "user", "content": user_msg})
    return prompts

async def query_openai_async(prompt, client: AsyncOpenAI, model, semaphore: Semaphore, max_retries=3):
    # Model-specific configurations
    if model.startswith("o"):
        timeout = 180
        temperature = 1.0
    else:
        timeout = 60
        temperature = 0.2
    
    async with semaphore:
        attempt = 0
        while attempt < max_retries:
            try:
                response = await asyncio.wait_for(
                    client.beta.chat.completions.parse(
                        model=model,
                        messages=prompt,
                        response_format=ErrorResponse,
                        temperature=temperature,
                    ),
                    timeout=timeout
                )

                # With beta.chat.completions.parse(), the response is already parsed
                parsed_data = response.choices[0].message.parsed
                return parsed_data.model_dump()

            except asyncio.TimeoutError:
                print(f"[{attempt}/{max_retries}] Timeout Error → Retry")
                await asyncio.sleep(20)  # Exponential backoff

            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for API usage/rate limit errors - infinite retry
                if any(keyword in error_msg for keyword in ["rate limit", "quota", "usage limit", "too many requests", "429", "server error"]):
                    print(f"[∞] Rate Limit/Quota Error: {e} → Infinite Retry")
                    wait_time = min(2 ** min(attempt, 6), 60)  # Cap exponential backoff
                    print(f"[∞] Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    # Don't increment attempt counter for rate limits - infinite retry
                    continue

                elif any(keyword in error_msg for keyword in ["parse", "json", "schema", "validation", "format"]):
                    print(f"[{attempt}/{max_retries}] JSON/Parse Error: {e} → Retry")
                else:
                    print(f"[{attempt}/{max_retries}] OpenAPI Error: {e} → Retry")
                await asyncio.sleep(20)  # Exponential backoff
                attempt += 1

        # If all retries fail, return empty structure
        return {"errors": [], "error": "Max retries exceeded"}

async def process_item(idx, row, client, model, sem, result_queue):
    result_row = row.to_dict()
    try:
        src = row['source_segment']
        tgt = row['hypothesis_segment']
        ref = row.get('reference_segment', None)
        src_lang = row['source_lang']
        tgt_lang = row['target_lang']
        prompt = build_prompt(src, src_lang, tgt, tgt_lang)
        result = await query_openai_async(prompt, client, model, sem)

        if (idx) % 200 == 0 and idx > 0:
            print(f"[{idx}] Evaluation completed")

        result_row["answer"] = result

    except Exception as e:
        print(f"[{idx}] Error processing item: {e}")
        result_row["answer"] = {"error": str(e)}

    finally:
        await result_queue.put(result_row)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True, help="Input TSV file path")
    parser.add_argument("--endpoint", required=True, help="OpenAI endpoint URL")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--api_version", default="2024-10-21", help="OpenAI API version")
    parser.add_argument("--deployment_name", required=True, help="OpenAI deployment name (model)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--worker", type=int, default=8, help="Number of parallel requests")
    parser.add_argument("--end", type=int, help="End index (exclusive). If not provided, process all sentences")
    parser.add_argument("--source_lang", help="Source language to filter (e.g., 'en'). If not provided, process all source languages")
    parser.add_argument("--target_lang", help="Target language to filter (e.g., 'de'). If not provided, process all target languages")
    args = parser.parse_args()

    # Read TSV file into DataFrame with original index
    df = pd.read_csv(args.input_tsv, sep='\t')
    df['original_index'] = df.index  # Preserve original order

    # Filter by source_lang and target_lang if specified
    if args.source_lang:
        df = df[df['source_lang'] == args.source_lang]
        print(f"[INFO] Filtered by source_lang: {args.source_lang} ({len(df)} sentences)")

    if args.target_lang:
        df = df[df['target_lang'] == args.target_lang]
        print(f"[INFO] Filtered by target_lang: {args.target_lang} ({len(df)} sentences)")

    # Apply end limit if specified
    if args.end is not None:
        df = df.iloc[:args.end]
        print(f"[INFO] Processing sentences 0 to {args.end-1} (total: {args.end} sentences)")
    else:
        print(f"[INFO] Processing all {len(df)} sentences")

    # Reset index for filtered data to have sequential progress tracking
    df = df.reset_index(drop=True)
    df['progress_index'] = df.index

    #  OpenAI client setup
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.endpoint + "/openai/v1",
        default_query= {
            "api-version": "preview"
        }
    )

    sem = Semaphore(args.worker)

    # Create thread-safe queue for results
    result_queue = asyncio.Queue()

    # Process items in parallel with error handling
    total_items = len(df)
    tasks = [
        process_item(row['progress_index'], row, client, args.deployment_name, sem, result_queue)
        for idx, row in df.iterrows()
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


    # Collect results from queue - wait for all expected results
    results = []
    expected_results = total_items
    
    # Wait for all results with timeout
    for i in range(expected_results):
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=5.0)
            results.append(result)
        except asyncio.TimeoutError:
            print(f"[WARNING] Timeout waiting for result {i+1}/{expected_results}")
            break

    if len(results) < expected_results:
        print(f"[WARNING] Only collected {len(results)} results out of {expected_results}. Some items may not have been processed successfully.")

    output_df = pd.DataFrame(results)

    # Verify we have the expected number of results
    if len(output_df) != expected_results:
        print(f"[WARNING] Expected {expected_results} results, but got {len(output_df)}")
    
    output_df = output_df.sort_values('original_index')
    output_df = output_df.drop(columns=['original_index'])

    if 'progress_index' in output_df.columns:
        output_df = output_df.drop(columns=['progress_index'])

    columns_to_remove = ['start_indices', 'end_indices', 'error_types']
    for col in columns_to_remove:
        if col in output_df.columns:
            output_df = output_df.drop(columns=[col])
    
    # Save to JSONL
    output_df.to_json(args.output, orient='records', lines=True, force_ascii=False)
    print(f"[INFO] Successfully saved {len(output_df)} results to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
