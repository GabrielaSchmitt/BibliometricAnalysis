import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
import rispy
import plotly.express as px
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set page config
st.set_page_config(page_title="Bibliometric Analysis Tool", layout="wide")

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader>div>div>button {background-color: #4CAF50; color: white;}
    .reportview-container .markdown-text-container {font-family: Arial, sans-serif;}
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Bibliometric Analysis Tool")
st.markdown("""
Upload RIS files from different sources, tag them, and analyze the bibliometric data with interactive visualizations.
""")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'tags' not in st.session_state:
    st.session_state.tags = {}

def safe_extract_year(date_val):
    """Safely extract year from date field handling various formats and types"""
    if pd.isna(date_val) or date_val is None:
        return None
    try:
        if isinstance(date_val, (int, float)):
            return int(date_val) if date_val > 1000 else None
        date_str = str(date_val)
        if len(date_str) >= 4 and date_str[:4].isdigit():
            return int(date_str[:4])
        return None
    except (ValueError, TypeError):
        return None

def process_ris_files(uploaded_files):
    """Process uploaded RIS files and return cleaned DataFrame"""
    all_entries = []
    for uploaded_file in uploaded_files:
        try:
            # Read RIS file with rispy
            entries = rispy.loads(uploaded_file.getvalue().decode('utf-8'))
            
            # Add tag to each entry
            tag = st.session_state.tags.get(uploaded_file.name, "Untagged")
            for entry in entries:
                entry['tag'] = tag
                entry['source_file'] = uploaded_file.name
            
            all_entries.extend(entries)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if not all_entries:
        st.warning("No valid entries found in the uploaded files.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_entries)
    
    # Basic cleaning and standardization
    df['title'] = df['title'].astype(str).str.strip().str.title()
    
    # Handle authors (convert to list if not already)
    df['authors'] = df['authors'].apply(
        lambda x: [a.strip() for a in x] if isinstance(x, list) 
        else [str(x).strip()] if pd.notna(x) 
        else []
    )
    
    # Handle keywords (convert to list if not already)
    df['keywords'] = df['keywords'].apply(
        lambda x: [k.strip().lower() for k in x] if isinstance(x, list)
        else [str(x).strip().lower()] if pd.notna(x)
        else []
    )
    
    # Extract year from date with robust handling
    df['year'] = df['date'].apply(safe_extract_year)
    
    # Handle citations (convert to numeric)
    df['citations'] = pd.to_numeric(df.get('citations', 0), errors='coerce').fillna(0)
    
    # Remove duplicates based on title and authors
    df['unique_id'] = df['title'] + df['authors'].apply(
        lambda x: ';'.join(sorted(x)) if x else ''
    )
    df = df.drop_duplicates(subset='unique_id', keep='first')
    
    return df

# File upload and tagging section
st.header("📂 Upload and Tag RIS Files")
uploaded_files = st.file_uploader("Upload RIS files", type=['ris'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"File: {uploaded_file.name}")
        with col2:
            tag = st.text_input(f"Tag for {uploaded_file.name}", key=f"tag_{uploaded_file.name}")
            if tag:
                st.session_state.tags[uploaded_file.name] = tag

    if st.button("Process Files"):
        with st.spinner("Processing files..."):
            st.session_state.data = process_ris_files(uploaded_files)
            if not st.session_state.data.empty:
                st.success(f"Processed {len(st.session_state.data)} unique articles from {len(uploaded_files)} files.")

# Analysis section
if not st.session_state.data.empty:
    df = st.session_state.data
    
    st.header("📊 Analysis Results")
    
    # 1. Total Articles by Database
    st.subheader("1. Total Articles by Database")
    if 'tag' in df.columns:
        fig1 = px.bar(df['tag'].value_counts().reset_index(), 
                     x='index', y='tag',
                     labels={'index': 'Database', 'tag': 'Number of Articles'},
                     color='index')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Tag information not available for analysis.")
    
    # 2. Most Relevant Article (most citations)
    st.subheader("2. Most Relevant Article (Most Citations)")
    if 'citations' in df.columns and not df['citations'].isnull().all():
        top_article = df.loc[df['citations'].idxmax()]
        st.markdown(f"""
        - **Title:** {top_article.get('title', 'N/A')}
        - **Authors:** {', '.join(top_article['authors']) if isinstance(top_article.get('authors'), list) else 'N/A'}
        - **Year:** {top_article.get('year', 'N/A')}
        - **Citations:** {int(top_article['citations'])}
        - **Source:** {top_article.get('tag', 'N/A')}
        """)
    else:
        st.warning("Citation data not available in the RIS files.")
    
    # 3. Publish or Perish (Articles by Year)
    st.subheader("3. Articles by Year (Publish or Perish)")
    if 'year' in df.columns and not df['year'].isnull().all():
        year_counts = df['year'].value_counts().sort_index()
        fig3 = px.line(year_counts.reset_index(), x='index', y='year',
                      labels={'index': 'Year', 'year': 'Number of Articles'},
                      markers=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Year information not available or incomplete in the RIS files.")
    
    # 4. Authors with Highest Participation
    st.subheader("4. Top Authors in the Portfolio")
    all_authors = [author for sublist in df['authors'].dropna() for author in sublist if author]
    if all_authors:
        top_authors = pd.Series(all_authors).value_counts().head(10)
        fig4 = px.bar(top_authors.reset_index(), 
                     x='index', y=0,
                     labels={'index': 'Author', '0': 'Number of Articles'})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Author information not available in the RIS files.")
    
    # 5. Most Used Keywords
    st.subheader("5. Most Used Keywords")
    all_keywords = [keyword for sublist in df['keywords'].dropna() for keyword in sublist if keyword]
    if all_keywords:
        # Clean keywords (remove very short words, punctuation, etc.)
        cleaned_keywords = [re.sub(r'[^\w\s]', '', k) for k in all_keywords if len(k) > 3]
        top_keywords = pd.Series(cleaned_keywords).value_counts().head(20)
        
        # Word cloud
        st.markdown("**Word Cloud of Keywords**")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(cleaned_keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Bar chart
        st.markdown("**Top Keywords**")
        fig5 = px.bar(top_keywords.reset_index(), 
                     x='index', y=0,
                     labels={'index': 'Keyword', '0': 'Frequency'})
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Keyword information not available in the RIS files.")
    
    # 6. Author Network Diagram
    st.subheader("6. Author Collaboration Network")
    if all_authors:
        try:
            # Create co-authorship network
            edges = []
            for authors in df['authors'].dropna():
                if len(authors) > 1:
                    # Remove empty author names if any
                    authors = [a for a in authors if a]
                    for i in range(len(authors)):
                        for j in range(i+1, len(authors)):
                            edges.append((authors[i], authors[j]))
            
            if edges:
                G = nx.Graph()
                G.add_edges_from(edges)
                
                # Get top 30 authors for visualization (for performance)
                top_authors_list = [author for author, _ in Counter(all_authors).most_common(30)]
                G_filtered = G.subgraph(top_authors_list)
                
                # Plot
                pos = nx.spring_layout(G_filtered, k=0.5, iterations=50)
                plt.figure(figsize=(12, 10))
                nx.draw_networkx_nodes(G_filtered, pos, node_size=100, alpha=0.8)
                nx.draw_networkx_edges(G_filtered, pos, width=1, alpha=0.2)
                nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_family='sans-serif')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.warning("Insufficient co-authorship data to create a network.")
        except Exception as e:
            st.error(f"Error creating author network: {str(e)}")
    else:
        st.warning("Author information not available in the RIS files.")
    
    # 7. Most Relevant Publishing Institution
    st.subheader("7. Top Publishing Institutions")
    if 'publisher' in df.columns and not df['publisher'].isnull().all():
        top_publishers = df['publisher'].value_counts().head(10)
        fig7 = px.bar(top_publishers.reset_index(), 
                     x='index', y='publisher',
                     labels={'index': 'Institution', 'publisher': 'Number of Articles'})
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("Publisher information not available in the RIS files.")
    
    # 8. Country with Most Publications
    st.subheader("8. Top Countries by Publications")
    if 'address' in df.columns and not df['address'].isnull().all():
        # Simple country extraction (this could be enhanced with proper geocoding)
        countries = df['address'].str.extract(r'([A-Za-z\s]+),?\s*[A-Z]{2,3}')[0].str.strip()
        if not countries.isnull().all():
            top_countries = countries.value_counts().head(10)
            fig8 = px.bar(top_countries.reset_index(), 
                         x='index', y='address',
                         labels={'index': 'Country', 'address': 'Number of Articles'})
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.warning("Could not extract country information from address fields.")
    else:
        st.warning("Address information not available in the RIS files.")
    
    # 9. Database Quality Ranking
    st.subheader("9. Database Quality Ranking")
    if 'tag' in df.columns and 'citations' in df.columns and not df['citations'].isnull().all():
        quality_metrics = df.groupby('tag').agg(
            avg_citations=('citations', 'mean'),
            median_citations=('citations', 'median'),
            top_10_percent=('citations', lambda x: np.percentile(x, 90)),
            total_articles=('citations', 'count')
        ).sort_values('avg_citations', ascending=False)
        
        st.dataframe(quality_metrics.style.background_gradient(cmap='Blues'))
        
        # Visualization
        fig9 = px.bar(quality_metrics.reset_index(), 
                     x='tag', y='avg_citations',
                     labels={'tag': 'Database', 'avg_citations': 'Average Citations'},
                     title='Database Quality by Average Citations')
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.warning("Citation data not available to calculate database quality metrics.")

# Download button for processed data
if not st.session_state.data.empty:
    st.download_button(
        label="Download Processed Data as CSV",
        data=st.session_state.data.to_csv(index=False).encode('utf-8'),
        file_name='bibliometric_analysis_results.csv',
        mime='text/csv'
    )