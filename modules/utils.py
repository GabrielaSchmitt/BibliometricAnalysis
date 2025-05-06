import pandas as pd
import numpy as np
import streamlit as st
import base64
import io
from typing import Dict, List, Any
import plotly.express as px

def get_colors() -> List[str]:
    """
    Return a consistent color palette for visualizations.
    
    Returns:
        List of color hex codes
    """
    return px.colors.qualitative.Plotly

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Create a download link for a DataFrame.
    
    Args:
        df: DataFrame to download
        filename: Name of the file
        
    Returns:
        HTML link for downloading the data
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
    return href

def generate_sample_data() -> pd.DataFrame:
    """
    Generate sample bibliometric data for demonstration purposes.
    
    Returns:
        DataFrame with sample data
    """
    # Generate sample data
    n_samples = 100
    
    # Sample databases
    databases = ['Scopus', 'Web of Science', 'IEEE Xplore', 'PubMed']
    db_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Sample authors (with some appearing multiple times)
    all_authors = [
        'Smith, John', 'Chen, Wei', 'Garcia, Maria', 'Kumar, Raj', 
        'Wang, Li', 'Johnson, Sarah', 'Kim, Min-Ji', 'Patel, Amit',
        'Anderson, Lisa', 'Rodriguez, Carlos', 'Zhang, Yan', 'Brown, Daniel',
        'Lee, Jung-Ho', 'Miller, Emma', 'Nguyen, Minh', 'Taylor, James'
    ]
    
    # Sample keywords
    all_keywords = [
        'machine learning', 'artificial intelligence', 'deep learning', 
        'neural networks', 'data science', 'big data', 'natural language processing',
        'computer vision', 'algorithms', 'data mining', 'robotics',
        'pattern recognition', 'image processing', 'information retrieval',
        'knowledge discovery', 'sentiment analysis', 'speech recognition',
        'computational linguistics', 'semantic web', 'reinforcement learning'
    ]
    
    # Sample countries
    all_countries = [
        'United States', 'China', 'United Kingdom', 'Germany', 'Canada',
        'Australia', 'France', 'Japan', 'Italy', 'India', 'South Korea',
        'Netherlands', 'Switzerland', 'Singapore', 'Spain'
    ]
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Generate publication year (weighted towards more recent years)
        year = np.random.choice(range(2010, 2025), p=[0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10, 0.11, 0.11, 0.06, 0.05])
        
        # Generate citation count (log-normal distribution to simulate real-world citation patterns)
        citation_count = int(np.random.lognormal(2, 1))
        
        # Select database source
        source_db = np.random.choice(databases, p=db_weights)
        
        # Generate between 1-5 authors
        num_authors = np.random.randint(1, 6)
        authors = np.random.choice(all_authors, size=num_authors, replace=False).tolist()
        
        # Generate between 3-8 keywords
        num_keywords = np.random.randint(3, 9)
        keywords = np.random.choice(all_keywords, size=num_keywords, replace=False).tolist()
        
        # Select institution and country
        institution = np.random.choice(all_institutions)
        country = np.random.choice(all_countries)
        
        # Create record
        record = {
            'title': f"Sample Article {i+1}: Research on {np.random.choice(keywords)}",
            'author': authors,
            'author_list': authors,
            'journal': f"Journal of {np.random.choice(['Science', 'Technology', 'Research', 'Data', 'Innovation'])} {np.random.randint(1, 50)}",
            'year': year,
            'abstract': f"This is a sample abstract for article {i+1} related to {', '.join(np.random.choice(keywords, 2))}.",
            'keywords': keywords,
            'keyword_list': keywords,
            'doi': f"10.1234/sample.{2023}.{i+1000}",
            'citation_count': citation_count,
            'institution': institution,
            'country': country,
            'primary_institution': institution,
            'primary_country': country,
            'source_db': source_db,
            'normalized_citations': citation_count / 100,  # Just a sample normalization
            'completeness': np.random.uniform(0.7, 1.0),
            'quality_score': np.random.uniform(0.5, 1.0)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)