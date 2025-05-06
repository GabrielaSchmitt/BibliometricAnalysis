import pandas as pd
import re
import streamlit as st
from typing import List, Dict, Any

def parse_ris_file(file_path: str, source_tag: str) -> pd.DataFrame:
    """
    Parse a single RIS file and convert it to a pandas DataFrame.
    
    Args:
        file_path: Path to the RIS file
        source_tag: Tag identifying the database source
        
    Returns:
        DataFrame containing the parsed RIS data
    """
    try:
        # Read the RIS file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception as e:
            st.error(f"Error reading file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    # Split the content into individual records
    # RIS files separate records with blank lines or ER tags
    records_raw = re.split(r'\nER\s+-\s*\n?', content)
    
    parsed_records = []
    
    for record in records_raw:
        if not record.strip():
            continue
            
        # Parse the record into a dictionary
        record_dict = parse_record(record)
        
        if record_dict:
            # Add the source tag
            record_dict['source_db'] = source_tag
            parsed_records.append(record_dict)
    
    # Convert to DataFrame
    if parsed_records:
        df = pd.DataFrame(parsed_records)
        return df
    else:
        return pd.DataFrame()

def parse_record(record_text: str) -> Dict[str, Any]:
    """
    Parse a single RIS record into a dictionary.
    
    Args:
        record_text: Text content of a single RIS record
        
    Returns:
        Dictionary with parsed fields
    """
    lines = record_text.strip().split('\n')
    record = {}
    
    # Common RIS tags mapping
    field_mapping = {
        'TI': 'title',
        'T1': 'title',
        'AU': 'author',
        'A1': 'author',
        'JO': 'journal',
        'JF': 'journal',
        'J1': 'journal',
        'JA': 'journal',
        'T2': 'journal',
        'PY': 'year',
        'Y1': 'year',
        'DA': 'date',
        'AB': 'abstract',
        'N1': 'notes',
        'N2': 'abstract',
        'KW': 'keywords',
        'UR': 'url',
        'DO': 'doi',
        'VL': 'volume',
        'IS': 'issue',
        'SP': 'start_page',
        'EP': 'end_page',
        'C1': 'institution',
        'AD': 'address',
        'PM': 'pmid',
        'LA': 'language',
        'PT': 'publication_type',
        'TC': 'citation_count',  # Web of Science specific
        'Z9': 'total_citations',  # Web of Science specific
        'PU': 'publisher',
        'SN': 'issn',
        'C2': 'country'
    }
    
    current_tag = None
    current_content = ""
    
    # Collect multiple entries for fields like authors and keywords
    multi_value_fields = {'author', 'keywords', 'institution', 'country'}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a tag (2 chars followed by space or dash)
        if re.match(r'^[A-Z][A-Z0-9]\s+-\s+', line) or re.match(r'^[A-Z][A-Z0-9]\s+', line):
            # Process the previous tag if it exists
            if current_tag and current_content:
                field_name = field_mapping.get(current_tag, current_tag.lower())
                
                if field_name in multi_value_fields:
                    if field_name in record:
                        if not isinstance(record[field_name], list):
                            record[field_name] = [record[field_name]]
                        record[field_name].append(current_content)
                    else:
                        record[field_name] = [current_content]
                else:
                    record[field_name] = current_content
            
            # Get the new tag and content
            match = re.match(r'^([A-Z][A-Z0-9])\s+-?\s+(.*)', line)
            if match:
                current_tag = match.group(1)
                current_content = match.group(2)
            else:
                current_tag = None
                current_content = ""
        elif current_tag:
            # Continuation of the current tag's content
            current_content += " " + line
    
    # Process the last tag
    if current_tag and current_content:
        field_name = field_mapping.get(current_tag, current_tag.lower())
        
        if field_name in multi_value_fields:
            if field_name in record:
                if not isinstance(record[field_name], list):
                    record[field_name] = [record[field_name]]
                record[field_name].append(current_content)
            else:
                record[field_name] = [current_content]
        else:
            record[field_name] = current_content
    
    # Post-processing
    if 'notes' in record:
        cited_match = re.search(r'Cited By: (\d+)', record['notes'])
        if cited_match:
            record['citation_count'] = int(cited_match.group(1))

    
    # Extract year from date if year is not present
    if 'year' not in record and 'date' in record:
        date_str = record['date']
        year_match = re.search(r'(\d{4})', date_str)
        if year_match:
            record['year'] = year_match.group(1)
    
    # Extract country from address if country is not present
    if 'country' not in record and 'address' in record:
        address = record['address']
        # Simple country extraction - last word in the address
        words = address.split(',')
        if words:
            potential_country = words[-1].strip()
            if len(potential_country) > 2:  # Avoid abbreviations
                record['country'] = [potential_country]
    
    # Convert citation count to integer if present
    if 'citation_count' in record:
        try:
            record['citation_count'] = int(record['citation_count'])
        except (ValueError, TypeError):
            pass
    elif 'total_citations' in record:
        try:
            record['citation_count'] = int(record['total_citations'])
            del record['total_citations']
        except (ValueError, TypeError):
            pass
    
    return record

def parse_ris_files(file_paths: List[str], source_tags: List[str]) -> pd.DataFrame:
    """
    Parse multiple RIS files and combine them into a single DataFrame.
    
    Args:
        file_paths: List of paths to RIS files
        source_tags: List of tags identifying the database source for each file
        
    Returns:
        Combined DataFrame containing all parsed RIS data
    """
    if len(file_paths) != len(source_tags):
        st.error("Number of file paths must match number of source tags")
        return pd.DataFrame()
    
    all_data = []
    
    for file_path, source_tag in zip(file_paths, source_tags):
        df = parse_ris_file(file_path, source_tag)
        if not df.empty:
            all_data.append(df)
    
    if all_data:
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()