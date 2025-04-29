import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from collections import Counter

def clean_title(title: str) -> str:
    """
    Clean and standardize article titles for better duplicate detection.
    
    Args:
        title: The title string to clean
        
    Returns:
        Cleaned title string
    """
    if not isinstance(title, str):
        return ""
    
    # Remove special characters and extra spaces
    cleaned = re.sub(r'[^\w\s]', '', title.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def clean_author_name(author: str) -> str:
    """
    Clean and standardize author names for better duplicate detection.
    
    Args:
        author: The author name to clean
        
    Returns:
        Cleaned author name
    """
    if not isinstance(author, str):
        return ""
    
    # Remove punctuation except periods in initials
    author = re.sub(r'[^\w\s\.]', '', author)
    
    # Handle different name formats:
    # Last, First -> First Last
    if ',' in author:
        parts = author.split(',', 1)
        if len(parts) == 2:
            last, first = parts
            author = f"{first.strip()} {last.strip()}"
    
    # Convert to lowercase and normalize spaces
    author = re.sub(r'\s+', ' ', author.lower()).strip()
    
    return author

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles from the dataset based on title and DOI similarity.
    
    Args:
        df: DataFrame containing bibliometric data
        
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Clean titles for better duplicate detection
    data['clean_title'] = data['title'].apply(clean_title)
    
    # Group by DOI if available
    if 'doi' in data.columns:
        # Clean DOI values
        data['clean_doi'] = data['doi'].str.lower().str.strip()
        
        # Group by DOI to find duplicates
        doi_groups = data.groupby('clean_doi')
        
        # Keep only the record with the most information for each DOI group
        unique_records = []
        
        for doi, group in doi_groups:
            if pd.isna(doi) or doi == '':
                # Skip empty DOIs
                unique_records.append(group)
            else:
                # For records with the same DOI, keep the one with the most complete data
                best_record = select_best_record(group)
                unique_records.append(pd.DataFrame([best_record]))
        
        data = pd.concat(unique_records)
    
    # Further deduplicate based on title similarity
    title_groups = {}
    
    for idx, row in data.iterrows():
        title = row['clean_title']
        if not title:
            continue
            
        found_match = False
        
        for group_title in list(title_groups.keys()):
            # Check for high similarity between titles
            if title_similarity(title, group_title) > 0.85:
                title_groups[group_title].append(row.to_dict())
                found_match = True
                break
        
        if not found_match:
            title_groups[title] = [row.to_dict()]
    
    # Select the best record from each title group
    unique_records = []
    
    for group in title_groups.values():
        if len(group) == 1:
            unique_records.append(group[0])
        else:
            best_record = select_best_record(pd.DataFrame(group))
            unique_records.append(best_record)
    
    # Create a new DataFrame with unique records
    # Convert lists to strings for proper DataFrame creation
    processed_records = []
    for record in unique_records:
        processed_record = {}
        for key, value in record.items():
            if isinstance(value, list):
                processed_record[key] = '; '.join(map(str, value))
            else:
                processed_record[key] = value
        processed_records.append(processed_record)
    
    unique_data = pd.DataFrame(processed_records)
    
    # Drop cleaning columns
    if 'clean_title' in unique_data.columns:
        unique_data = unique_data.drop('clean_title', axis=1)
    if 'clean_doi' in unique_data.columns:
        unique_data = unique_data.drop('clean_doi', axis=1)
    
    return unique_data

def title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles using Jaccard similarity.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Similarity score between 0 and 1
    """
    if not title1 or not title2:
        return 0
    
    # Use sets of words for Jaccard similarity
    words1 = set(title1.split())
    words2 = set(title2.split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    return intersection / union

def select_best_record(group: pd.DataFrame) -> Dict[str, Any]:
    """
    Select the most complete record from a group of duplicate records.
    
    Args:
        group: DataFrame containing duplicate records
        
    Returns:
        Dictionary representing the best record
    """
    if len(group) == 1:
        return group.iloc[0].to_dict()
    
    # Prioritize records with more complete information
    # Calculate the number of non-null values in each record
    completeness = group.notna().sum(axis=1)
    
    # Prioritize records with citation counts if available
    if 'citation_count' in group.columns:
        has_citations = ~group['citation_count'].isna()
        
        if has_citations.any():
            # Sort by completeness and citation count
            group = group[has_citations]
            
            # If multiple records have citation counts, take the one with highest value
            highest_citation_idx = group['citation_count'].fillna(0).idxmax()
            return group.loc[highest_citation_idx].to_dict()
    
    # Otherwise, take the most complete record
    most_complete_idx = completeness.idxmax()
    return group.loc[most_complete_idx].to_dict()

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the bibliometric data for analysis.
    
    Args:
        df: DataFrame containing bibliometric data with duplicates removed
        
    Returns:
        Processed DataFrame ready for analysis
    """
    if df.empty:
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Ensure all columns exist with default values
    required_columns = [
        'title', 'author', 'journal', 'year', 'abstract', 
        'keywords', 'doi', 'citation_count', 'institution', 'country'
    ]
    
    for col in required_columns:
        if col not in data.columns:
            data[col] = None
    
    # Process citation count
    if 'citation_count' in data.columns:
        data['citation_count'] = pd.to_numeric(data['citation_count'], errors='coerce').fillna(0).astype(int)
    
    # Process year
    if 'year' in data.columns:
        data['year'] = pd.to_numeric(data['year'], errors='coerce')
        # Fill missing years with a reasonable default
        data['year'] = data['year'].fillna(2020)
        data['year'] = data['year'].astype(int)
    
    # Process authors: normalize author lists
    def normalize_authors(authors):
        if isinstance(authors, list):
            return [clean_author_name(author) for author in authors if author]
        elif isinstance(authors, str):
            # Split by semicolon if present
            if ';' in authors:
                return [clean_author_name(a.strip()) for a in authors.split(';')]
            return [clean_author_name(authors)]
        else:
            return []
    
    data['author_list'] = data['author'].apply(normalize_authors)
    
    # Process keywords: normalize keyword lists
    def normalize_keywords(keywords):
        if isinstance(keywords, list):
            return [kw.lower().strip() for kw in keywords if kw]
        elif isinstance(keywords, str):
            # Some RIS files may have keywords separated by semicolons or commas
            kw_list = re.split(r'[;,]', keywords)
            return [kw.lower().strip() for kw in kw_list if kw.strip()]
        else:
            return []
    
    data['keyword_list'] = data['keywords'].apply(normalize_keywords)
    
    # Process institutions
    def normalize_institutions(institutions):
        if isinstance(institutions, list):
            return [inst.strip() for inst in institutions if inst]
        elif isinstance(institutions, str):
            # Split by semicolon if present
            if ';' in institutions:
                return [inst.strip() for inst in institutions.split(';')]
            return [institutions.strip()]
        else:
            return []
    
    data['institution_list'] = data['institution'].apply(normalize_institutions)
    
    # Process countries
    def normalize_countries(countries):
        if isinstance(countries, list):
            return [country.strip() for country in countries if country]
        elif isinstance(countries, str):
            # Split by semicolon if present
            if ';' in countries:
                return [country.strip() for country in countries.split(';')]
            return [countries.strip()]
        else:
            return []
    
    data['country_list'] = data['country'].apply(normalize_countries)
    
    # For institutions/countries with multiple entries, use the most frequent one
    def get_primary_institution(inst_list):
        if not inst_list:
            return None
        counter = Counter(inst_list)
        return counter.most_common(1)[0][0]
    
    def get_primary_country(country_list):
        if not country_list:
            return None
        counter = Counter(country_list)
        return counter.most_common(1)[0][0]
    
    data['primary_institution'] = data['institution_list'].apply(get_primary_institution)
    data['primary_country'] = data['country_list'].apply(get_primary_country)
    
    # Calculate additional metrics
    
    # Quality score based on citation count and completeness
    # Normalize citation count to 0-1 range
    if 'citation_count' in data.columns and not data['citation_count'].isna().all():
        max_citations = data['citation_count'].max()
        if max_citations > 0:
            data['normalized_citations'] = data['citation_count'] / max_citations
        else:
            data['normalized_citations'] = 0
    else:
        data['normalized_citations'] = 0
    
    # Completeness score (percentage of non-null values in important columns)
    important_cols = ['title', 'author', 'journal', 'year', 'abstract', 'keywords', 'doi']
    data['completeness'] = data[important_cols].notna().sum(axis=1) / len(important_cols)
    
    # Combined quality score
    data['quality_score'] = (data['normalized_citations'] * 0.7) + (data['completeness'] * 0.3)
    
    return data