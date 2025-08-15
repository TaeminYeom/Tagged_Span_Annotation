#!/usr/bin/env python3

import json
import csv
import argparse
from pathlib import Path


def find_error_spans(hypothesis_segment, error_text):
    """
    Returns:
        tuple: (start_index, end_index) or None if not found
    """
    if not error_text or type(error_text) != str or not hypothesis_segment:
        return None

    start_idx = hypothesis_segment.find(error_text)
    if start_idx != -1:
        end_idx = start_idx + len(error_text)  # exclusive end index
        return (start_idx, end_idx)
    
    return None


def parse_annotation_tags(annotation):
    """
    Returns:
        list: (error_number, start_index, end_index, error_text) tuple's list
    """
    import re
    
    error_spans = []
    # <v>...</v> or <vN>...</vN> pattern
    pattern = r'<v(\d*)>(.*?)</v\1>'
    for match in re.finditer(pattern, annotation):
        error_num_str = match.group(1)
        error_num = int(error_num_str) if error_num_str else 0  # default to 0 if no number
        error_text = match.group(2)

        # Compute the total length of tags before this match (to adjust indices)
        before_text = annotation[:match.start()]
        tag_length_before = sum(len(tag) for tag in re.findall(r'<v\d*>|</v\d*>', before_text))

        # Calculate actual start position in clean text (excluding tag lengths)
        start_in_clean = match.start() - tag_length_before
        end_in_clean = start_in_clean + len(error_text)  # exclusive end index

        error_spans.append((error_num, start_in_clean, end_in_clean, error_text))
    
    return error_spans


def parse_annotation_format(answer_data):
    """
    parse annotation + errors answer_data format

    Args:
        answer_data (dict): including annotated_translation and errors keys

    Returns:
        dict: parsed error information with start_indices, end_indices, error_types, categories
    """
    # Support both 'annotation' and 'annotated_translation' keys
    annotation = answer_data.get('annotation', '') or answer_data.get('annotated_translation', '')
    errors = answer_data.get('errors', [])
    
    # Extract tag span info from annotation
    error_spans = parse_annotation_tags(annotation)
    
    start_indices = []
    end_indices = []
    error_types = []
    categories = []
    
    # if errors is a dict
    if isinstance(errors, dict):
        sorted_keys = sorted(errors.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for key in sorted_keys:
            error = errors[key]
            key_num = int(key) if key.isdigit() else 0
            
            if isinstance(error, dict) and key_num < len(error_spans):
                _, start_idx, end_idx, error_text = error_spans[key_num]
                
                severity = error.get('severity', '').lower()
                category = error.get('category', '')
                
                if severity and category and "no-error" not in category:
                    start_indices.append(str(start_idx))
                    end_indices.append(str(end_idx))
                    error_types.append(severity)
                    categories.append(category)
    
    # if errors is a list
    elif isinstance(errors, list):
        for i, error in enumerate(errors):
            if isinstance(error, dict):
                if 'severity' in error and 'category' in error and i < len(error_spans):
                    _, start_idx, end_idx, error_text = error_spans[i]
                    
                    severity = error.get('severity', '').lower()
                    category = error.get('category', '')
                    
                    if severity and category and "no-error" not in category:
                        # default severity
                        if severity not in ["major", "minor"]:
                            severity = "minor"
                        start_indices.append(str(start_idx))
                        end_indices.append(str(end_idx))
                        error_types.append(severity)
                        categories.append(category)
                
                # {"Major": "category"} or {"Minor": "category"}
                elif i < len(error_spans):
                    _, start_idx, end_idx, error_text = error_spans[i]
                    
                    for severity_key, category in error.items():
                        severity = severity_key.lower()
                        if severity in ["major", "minor"] and "no-error" not in category:
                            start_indices.append(str(start_idx))
                            end_indices.append(str(end_idx))
                            error_types.append(severity)
                            categories.append(category)
                            break
    
    # If there is no error
    if not start_indices:
        return {
            'start_indices': '-1',
            'end_indices': '-1',
            'error_types': 'no-error',
            'categories': 'no-error'
        }
    
    return {
        'start_indices': ' '.join(start_indices),
        'end_indices': ' '.join(end_indices),
        'error_types': ' '.join(error_types),
        'categories': ' '.join(categories)
    }


def parse_errors_array_format(errors, hypothesis_segment):
    # {"text": "error_text", "severity": "Major/Minor", "category": "category"}
    # {"start": 53, "end": 53, "severity": "Major", "category": "accuracy/omission"}
    
    start_indices = []
    end_indices = []
    error_types = []
    categories = []
    
    for error in errors:
        if isinstance(error, dict):
            severity = error.get('severity', '').lower()
            category = error.get('category', '')
            
            if severity in ['major']:
                severity = 'major'
            elif severity in ['minor']:
                severity = 'minor'
            else:
                severity = 'minor'
            
            if category and "no-error" not in category:
                # Direct-index
                if 'start' in error and 'end' in error:
                    start_idx = error['start']
                    end_idx = error['end']
                    
                    # marks missing at omission
                    if "omission" in category.lower() and start_idx == end_idx:
                        start_indices.append("missing")
                        end_indices.append("missing")
                    else:
                        start_indices.append(str(start_idx))
                        end_indices.append(str(end_idx))
                    
                    error_types.append(severity)
                    categories.append(category)
                
                # GEMBA
                elif 'text' in error:
                    error_text = error.get('text', '')
                    if error_text:
                        # parse error spans from hypothesis_segment
                        span = find_error_spans(hypothesis_segment, error_text)
                        if span:
                            start_indices.append(str(span[0]))
                            end_indices.append(str(span[1]))
                            error_types.append(severity)
                            categories.append(category)
                        elif "omission" in category.lower():
                            # marks missing at omission
                            start_indices.append("missing")
                            end_indices.append("missing")
                            error_types.append(severity)
                            categories.append(category)
    
    # If there is no error
    if not start_indices:
        return {
            'start_indices': '-1',
            'end_indices': '-1',
            'error_types': 'no-error',
            'categories': 'no-error'
        }
    
    return {
        'start_indices': ' '.join(start_indices),
        'end_indices': ' '.join(end_indices),
        'error_types': ' '.join(error_types),
        'categories': ' '.join(categories)
    }


def parse_answer_field(answer_data, hypothesis_segment):
    # answer_data is not properly formatted
    if not isinstance(answer_data, dict):
        return {
            'start_indices': '-1',
            'end_indices': '-1',
            'error_types': 'no-error',
            'categories': 'no-error'
        }

    # annotated_translation
    if ('annotation' in answer_data or 'annotated_translation' in answer_data) and 'errors' in answer_data:
        return parse_annotation_format(answer_data)
    
    # errors array (includes text, severity, category)
    if 'errors' in answer_data and isinstance(answer_data['errors'], list):
        return parse_errors_array_format(answer_data['errors'], hypothesis_segment)
    
    # Handle legacy formats
    start_indices = []
    end_indices = []
    error_types = []
    categories = []
    
    # Handle Major errors
    if 'Major' in answer_data and isinstance(answer_data['Major'], list):
        for error in answer_data['Major']:
            if isinstance(error, dict):
                error_types.append('major')
                # {"start": 134, "end": 152, "category": "accuracy/addition"}
                if 'category' in error and 'start' in error and 'end' in error:
                    if "no-error" not in error['category']:
                        categories.append(error['category'])
                        start_indices.append(str(error['start']))
                        end_indices.append(str(error['end']))
                else:
                    for category, error_text in error.items():
                        if "no-error" in category:
                            continue
                        categories.append(category)
                        span = find_error_spans(hypothesis_segment, error_text)
                        if span:
                            start_indices.append(str(span[0]))
                            end_indices.append(str(span[1]))
                        elif "omission" in category:
                            start_indices.append("missing")
                            end_indices.append("missing")

    # Handle Minor errors
    if 'Minor' in answer_data and isinstance(answer_data['Minor'], list):
        for error in answer_data['Minor']:
            if isinstance(error, dict):
                error_types.append('minor')
                # {"start": 134, "end": 152, "category": "accuracy/addition"}
                if 'category' in error and 'start' in error and 'end' in error:
                    if "no-error" not in error['category']:
                        categories.append(error['category'])
                        start_indices.append(str(error['start']))
                        end_indices.append(str(error['end']))
                else:
                    for category, error_text in error.items():
                        if "no-error" in category:
                            continue
                        categories.append(category)
                        span = find_error_spans(hypothesis_segment, error_text)
                        if span:
                            start_indices.append(str(span[0]))
                            end_indices.append(str(span[1]))
                        elif "omission" in category:
                            start_indices.append("missing")
                            end_indices.append("missing")

    # There is no error
    if not start_indices:
        return {
            'start_indices': '-1',
            'end_indices': '-1',
            'error_types': 'no-error',
            'categories': 'no-error'
        }
    
    return {
        'start_indices': ' '.join(start_indices),
        'end_indices': ' '.join(end_indices),
        'error_types': ' '.join(error_types),
        'categories': ' '.join(categories)
    }

def convert_jsonl_to_tsv(input_file, output_file=None):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Auto-generate output filename if not provided
    if output_file is None:
        output_file = input_path.with_suffix('.tsv')
    
    output_path = Path(output_file)
    
    # Read JSONL and build rows for TSV
    data = []
    headers = set()
    
    # Statistics counters
    total_error_spans = 0
    total_error_chars = 0
        
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                json_obj = json.loads(line)
                
                # Process 'answer' field
                error_info = {
                    'start_indices': '-1',
                    'end_indices': '-1',
                    'error_types': 'no-error',
                    'categories': 'no-error'
                }

                if 'answer' in json_obj:
                    answer_data = json_obj['answer']
                    hypothesis_segment = json_obj.get('hypothesis_segment', '')
                    error_info = parse_answer_field(answer_data, hypothesis_segment)
                    del json_obj['answer']  # remove raw 'answer' field
                    
                    # Update error counts
                    if error_info['start_indices'] != '-1':
                        start_indices = error_info['start_indices'].split()
                        end_indices = error_info['end_indices'].split()
                        
                        # Count spans
                        total_error_spans += len(start_indices)
                        
                        # Count characters inside spans
                        for start_str, end_str in zip(start_indices, end_indices):
                            if start_str != 'missing' and end_str != 'missing':
                                try:
                                    start_idx = int(start_str)
                                    end_idx = int(end_str)
                                    total_error_chars += (end_idx - start_idx)
                                except ValueError:
                                    continue

                json_obj.update(error_info)

                data.append(json_obj)
                headers.update(json_obj.keys())
            except json.JSONDecodeError as e:
                print(f"{line_num} : JSON Parsing Error : {e}")
                continue

    # Predefined column order for output
    specified_headers = [
        'doc_id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id',
        'source_segment', 'hypothesis_segment', 'reference_segment', 'domain_name',
        'method', 'start_indices', 'end_indices', 'error_types', 'categories'
    ]

    print(f"{len(data)} records processed.")

    # Write TSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=specified_headers, delimiter='\t', 
                               quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for row in data:
            # Normalize row to specified field order
            normalized_row = {}
            for header in specified_headers:
                # Use existing value or empty string
                normalized_row[header] = row.get(header, '')
            writer.writerow(normalized_row)
    
    print(f"Converting complete: {output_file}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    convert_jsonl_to_tsv(args.input, args.output)

if __name__ == "__main__":
    main()
