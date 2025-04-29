import streamlit as st
import pandas as pd
import os
import tempfile
from modules.ris_parser import parse_ris_files
from modules.data_processor import remove_duplicates, process_data
from modules.visualizations import (
    plot_articles_by_database,
    plot_most_cited_articles,
    plot_publish_or_perish,
    plot_top_authors,
    plot_keyword_cloud,
    plot_author_network,
    plot_top_institutions,
    plot_countries,
    plot_database_quality,
)
from modules.utils import get_colors, create_download_link

# Page configuration
st.set_page_config(
    page_title="Bibliometric Analysis Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.05rem;
        line-height: 1.5;
    }
    .st-emotion-cache-1r6slb0 {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# App header
st.title("📚 Bibliometric Analysis Tool")
st.markdown("Upload RIS files from different sources, tag them, and analyze your bibliometric portfolio.")

# Sidebar for file uploads and tagging
with st.sidebar:
    st.header("Data Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload RIS files", 
        type=["ris"], 
        accept_multiple_files=True,
        help="Upload RIS files from different bibliometric databases"
    )
    
    # Tag creation for uploaded files
    if uploaded_files:
        st.subheader("Tag Your Files")
        st.markdown("Assign a database tag to each uploaded file")
        
        for uploaded_file in uploaded_files:
            file_key = uploaded_file.name
            
            if file_key not in st.session_state.uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ris') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                st.session_state.uploaded_files[file_key] = {
                    'path': temp_path,
                    'tag': None
                }
            
            tag = st.text_input(
                f"Tag for {file_key}", 
                value=st.session_state.uploaded_files[file_key].get('tag', ''),
                key=f"tag_{file_key}",
                help="Enter the name of the database this file came from (e.g., Scopus, Web of Science)"
            )
            
            if tag:
                st.session_state.uploaded_files[file_key]['tag'] = tag
    
    # Process button
    process_btn = st.button(
        "Process and Analyze", 
        type="primary",
        help="Parse RIS files, remove duplicates, and generate visualizations",
        disabled=not all(file.get('tag') for file in st.session_state.uploaded_files.values()) or len(st.session_state.uploaded_files) == 0
    )
    
    if process_btn:
        # Reset processed state
        st.session_state.processed = False
        
        with st.spinner("Parsing RIS files..."):
            # Get file paths and tags
            file_paths = [file['path'] for file in st.session_state.uploaded_files.values()]
            tags = [file['tag'] for file in st.session_state.uploaded_files.values()]
            
            # Parse RIS files
            raw_data = parse_ris_files(file_paths, tags)
            
            if raw_data.empty:
                st.error("No valid data found in the uploaded RIS files. Please check the file format.")
            else:
                with st.spinner("Removing duplicates and processing data..."):
                    # Remove duplicates and process data
                    unique_data = remove_duplicates(raw_data)
                    processed_data = process_data(unique_data)
                    
                    # Store processed data in session state
                    st.session_state.data = processed_data
                    st.session_state.processed = True
                    
                    st.success(f"Data processed successfully! {len(processed_data)} unique articles found.")

    # Add download button for processed data if available
    if st.session_state.processed and st.session_state.data is not None:
        st.download_button(
            label="Download Processed Data (CSV)",
            data=st.session_state.data.to_csv(index=False).encode('utf-8'),
            file_name="processed_bibliometric_data.csv",
            mime="text/csv"
        )

# Main content area - Analysis results
if st.session_state.processed and st.session_state.data is not None:
    data = st.session_state.data
    
    # Overview metrics
    st.header("Bibliometric Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Articles", 
            value=len(data)
        )
    
    with col2:
        st.metric(
            label="Unique Authors", 
            value=data['author'].nunique()
        )
    
    with col3:
        st.metric(
            label="Date Range", 
            value=f"{data['year'].min()}-{data['year'].max()}"
        )
    
    with col4:
        st.metric(
            label="Sources/Journals", 
            value=data['journal'].nunique()
        )
    
    # Create tab sections for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Publication Analytics", "👥 Author Analytics", "🌐 Source & Geographic Analytics"])
    
    with tab1:
        st.subheader("Publication Metrics")
        
        # Total articles by database
        st.markdown("### Total Articles by Database")
        fig_by_db = plot_articles_by_database(data)
        st.plotly_chart(fig_by_db, use_container_width=True)
        
        # Most cited articles
        st.markdown("### Most Cited Articles")
        fig_most_cited = plot_most_cited_articles(data)
        st.plotly_chart(fig_most_cited, use_container_width=True)
        
        # Publish or Perish graph
        st.markdown("### Publications Over Time")
        fig_pub_perish = plot_publish_or_perish(data)
        st.plotly_chart(fig_pub_perish, use_container_width=True)
    
    with tab2:
        st.subheader("Author Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top authors
            st.markdown("### Authors with Most Publications")
            fig_top_authors = plot_top_authors(data)
            st.plotly_chart(fig_top_authors, use_container_width=True)
        
        with col2:
            # Most used keywords
            st.markdown("### Most Used Keywords")
            fig_keywords = plot_keyword_cloud(data)
            st.pyplot(fig_keywords)
        
        # Author network
        st.markdown("### Author Collaboration Network")
        st.info("This network diagram shows collaboration relationships between authors. Larger nodes indicate authors with more publications, and connected nodes represent co-authorship.")
        fig_network = plot_author_network(data)
        st.plotly_chart(fig_network, use_container_width=True)
    
    with tab3:
        st.subheader("Source & Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top institutions
            st.markdown("### Most Relevant Publication Institutions")
            fig_institutions = plot_top_institutions(data)
            st.plotly_chart(fig_institutions, use_container_width=True)
        
        with col2:
            # Countries with most publications
            st.markdown("### Countries with Most Publications")
            fig_countries = plot_countries(data)
            st.plotly_chart(fig_countries, use_container_width=True)
        
        # Database quality ranking
        st.markdown("### Database Quality Ranking")
        st.info("This graph ranks each database source by the average citation count of articles, providing insight into which database yielded higher quality research papers.")
        fig_db_quality = plot_database_quality(data)
        st.plotly_chart(fig_db_quality, use_container_width=True)

else:
    # Instructions for first-time users
    st.info("👈 Please upload your RIS files in the sidebar, tag them with their database sources, and click 'Process and Analyze' to start.")
    
    st.markdown("""
    ### How to Use This Tool:
    
    1. **Upload RIS Files**: Click the 'Upload RIS files' button in the sidebar and select one or more RIS files from your bibliometric searches.
    
    2. **Tag Your Files**: For each uploaded file, provide a tag to identify which database it came from (e.g., "Scopus", "Web of Science", "IEEE Xplore").
    
    3. **Process and Analyze**: Click the 'Process and Analyze' button to parse the files, remove duplicates, and generate visualizations.
    
    4. **Explore Results**: Review the generated visualizations to gain insights about your bibliometric data.
    
    5. **Download Processed Data**: After processing, you can download the cleaned and processed data as a CSV file.
    
    ### What are RIS Files?
    
    RIS is a standardized tag format developed by Research Information Systems for bibliographic citations. Most academic databases allow you to export search results in RIS format.
    """)
    
    # Sample visualizations for demonstration when no data is uploaded
    if st.checkbox("Show sample visualizations"):
        st.markdown("### Sample Visualizations (for demonstration only)")
        st.warning("These are sample visualizations using mock data. Upload your RIS files to see actual results.")
        
        # Generate sample data
        from modules.utils import generate_sample_data
        sample_data = generate_sample_data()
        
        tab1, tab2 = st.tabs(["Sample Charts 1", "Sample Charts 2"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sample1 = plot_articles_by_database(sample_data)
                st.plotly_chart(fig_sample1, use_container_width=True)
            
            with col2:
                fig_sample2 = plot_top_authors(sample_data)
                st.plotly_chart(fig_sample2, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sample3 = plot_publish_or_perish(sample_data)
                st.plotly_chart(fig_sample3, use_container_width=True)
            
            with col2:
                fig_sample4 = plot_countries(sample_data)
                st.plotly_chart(fig_sample4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Bibliometric Analysis Tool | Created with Streamlit | 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Clean up temporary files when the app is closed
def cleanup():
    for file_info in st.session_state.uploaded_files.values():
        try:
            os.remove(file_info['path'])
        except:
            pass

# Register the cleanup function to be called when the session ends
import atexit
atexit.register(cleanup)