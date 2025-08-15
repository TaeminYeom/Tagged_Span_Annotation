#!/usr/bin/env python3
"""
Extract source, hypothesis, and reference segments from TSV file for specific language pairs.
"""

import argparse
import pandas as pd
import random
from pathlib import Path


def extract_from_tsv(
    input_tsv: str,
    langs: str,
    output_tsv: str = None,
    output_src: str = None,
    output_hyp: str = None,
    output_ref: str = None,
    sample: bool = False,
    sample_size: int = 2000,
    random_seed: int = 42
):
    """
    Extract segments from TSV file for specific language pair.
    
    Args:
        input_tsv: Path to input TSV file
        langs: Language pair in format "en-cs"
        output_tsv: Path to output TSV file (optional)
        output_src: Path to output source text file (optional)
        output_hyp: Path to output hypothesis text file (optional)
        output_ref: Path to output reference text file (optional)
        sample: Whether to sample data
        sample_size: Number of samples to extract
        random_seed: Random seed for sampling
    """
    
    # Parse language pair
    source_lang, target_lang = langs.split('-')
    
    print(f"Reading TSV file: {input_tsv}")
    
    # Read TSV file
    df = pd.read_csv(input_tsv, sep='\t')
    
    print(f"Total rows in input file: {len(df)}")
    
    # Filter by language pair
    filtered_df = df[
        (df['source_lang'] == source_lang) & 
        (df['target_lang'] == target_lang)
    ]
    
    print(f"Rows for {langs}: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"No data found for language pair: {langs}")
        return
    
    # Sample data if requested
    if sample:
        if len(filtered_df) > sample_size:
            print(f"Sampling {sample_size} rows from {len(filtered_df)} available rows")
            random.seed(random_seed)
            filtered_df = filtered_df.sample(n=sample_size, random_state=random_seed)
        else:
            print(f"Using all {len(filtered_df)} rows (less than sample size {sample_size})")
    
    # Sort by doc_id and segment_id for consistency
    filtered_df = filtered_df.sort_values(['doc_id', 'segment_id'])
    
    # Write output TSV file
    if output_tsv:
        print(f"Writing TSV to: {output_tsv}")
        Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_tsv, sep='\t', index=False)
    
    # Write source segments
    if output_src:
        print(f"Writing source segments to: {output_src}")
        Path(output_src).parent.mkdir(parents=True, exist_ok=True)
        with open(output_src, 'w', encoding='utf-8') as f:
            for segment in filtered_df['source_segment']:
                f.write(f"{segment}\n")
    
    # Write hypothesis segments
    if output_hyp:
        print(f"Writing hypothesis segments to: {output_hyp}")
        Path(output_hyp).parent.mkdir(parents=True, exist_ok=True)
        with open(output_hyp, 'w', encoding='utf-8') as f:
            for segment in filtered_df['hypothesis_segment']:
                f.write(f"{segment}\n")
    
    # Write reference segments
    if output_ref:
        print(f"Writing reference segments to: {output_ref}")
        Path(output_ref).parent.mkdir(parents=True, exist_ok=True)
        with open(output_ref, 'w', encoding='utf-8') as f:
            for segment in filtered_df['reference_segment']:
                f.write(f"{segment}\n")
    
    print(f"Extraction complete. Final dataset size: {len(filtered_df)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Extract source, hypothesis, and reference segments from TSV file"
    )
    
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="Path to input TSV file"
    )
    
    parser.add_argument(
        "--langs",
        required=True,
        help="Language pair in format 'en-cs'"
    )
    
    parser.add_argument(
        "--output_tsv",
        help="Path to output TSV file"
    )
    
    parser.add_argument(
        "--output_src",
        help="Path to output source text file"
    )
    
    parser.add_argument(
        "--output_hyp",
        help="Path to output hypothesis text file"
    )
    
    parser.add_argument(
        "--output_ref",
        help="Path to output reference text file"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample data instead of using all data"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of samples to extract when --sample is used (default: 2000)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    extract_from_tsv(
        input_tsv=args.input_tsv,
        langs=args.langs,
        output_tsv=args.output_tsv,
        output_src=args.output_src,
        output_hyp=args.output_hyp,
        output_ref=args.output_ref,
        sample=args.sample,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
