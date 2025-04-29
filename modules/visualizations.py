import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from typing import List, Dict, Any, Tuple
import re
from itertools import combinations

# Set a consistent color palette for all charts
COLORS = px.colors.qualitative.Plotly

def plot_articles_by_database(data: pd.DataFrame):
    """
    Create a horizontal bar chart showing the number of articles per database.
    
    Args:
        data: Processed bibliometric DataFrame
        
    Returns:
        Plotly figure
    """
    if data.empty or 'source_db' not in data.columns:
        # Generate sample data if no real data
        source_counts = pd.Series({
            'Scopus': 35, 
            'Web of Science': 28, 
            'IEEE Xplore': 20, 
            'PubMed': 15
        })
    else:
        source_counts = data['source_db'].value_counts()
    
    # Sort by count in descending order
    source_counts = source_counts.sort_values(ascending=True)
    
    fig = px.bar(
        x=source_counts.values,
        y=source_counts.index,
        orientation='h',
        labels={'x': 'Number of Articles', 'y': 'Database'},
        title='Articles by Database',
        color=source_counts.index,
        color_discrete_sequence=COLORS,
        text=source_counts.values
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        showlegend=False,
        xaxis_title='Number of Articles',
        yaxis_title='Database',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def plot_most_cited_articles(data: pd.DataFrame, top_n: int = 10):
    """
    Create a horizontal bar chart showing the most cited articles.
    
    Args:
        data: Processed bibliometric DataFrame
        top_n: Number of top articles to show
        
    Returns:
        Plotly figure
    """
    if data.empty or 'citation_count' not in data.columns:
        # Generate sample data if no real data
        sample_data = pd.DataFrame({
            'title': [f'Sample Article {i+1}' for i in range(10)],
            'citation_count': [100, 85, 72, 65, 58, 52, 48, 45, 40, 35],
            'author_list': [['Author A', 'Author B'] for _ in range(10)]
        })
        plot_data = sample_data.sort_values('citation_count', ascending=False).head(top_n)
    else:
        # Use real data
        plot_data = data.sort_values('citation_count', ascending=False).head(top_n)
    
    # Prepare titles and authors for display
    def format_title(row):
        """Format title and authors for display."""
        title = row['title']
        if isinstance(title, str) and len(title) > 60:
            title = title[:57] + '...'
        
        authors = row.get('author_list', [])
        if authors and isinstance(authors, list):
            if len(authors) > 1:
                authors_str = f"{authors[0]} et al."
            else:
                authors_str = authors[0]
        else:
            authors_str = "Unknown"
        
        return f"{title}<br><i>{authors_str}</i>"
    
    # Apply the formatting function
    if 'author_list' in plot_data.columns:
        plot_data['formatted_title'] = plot_data.apply(format_title, axis=1)
    else:
        plot_data['formatted_title'] = plot_data['title'].apply(lambda x: x[:57] + '...' if len(x) > 60 else x)
    
    # Sort by citation count in ascending order for the horizontal bar chart
    plot_data = plot_data.sort_values('citation_count', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        plot_data,
        x='citation_count', 
        y='formatted_title',
        orientation='h',
        labels={'citation_count': 'Citation Count', 'formatted_title': ''},
        title=f'Top {top_n} Most Cited Articles',
        color='citation_count',
        color_continuous_scale='Viridis',
        text='citation_count'
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        xaxis_title='Citation Count',
        yaxis_title='',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
    )
    
    return fig

def plot_publish_or_perish(data: pd.DataFrame):
    """
    Create a line chart showing publications over time with a trend line.
    
    Args:
        data: Processed bibliometric DataFrame
        
    Returns:
        Plotly figure
    """
    if data.empty or 'year' not in data.columns:
        # Generate sample data if no real data
        years = list(range(2010, 2025))
        publications = [5, 8, 12, 15, 20, 25, 28, 32, 38, 42, 45, 48, 50, 47, 45]
        cumulative = np.cumsum(publications)
        
        plot_data = pd.DataFrame({
            'year': years,
            'publications': publications,
            'cumulative': cumulative
        })
    else:
        # Use real data
        yearly_counts = data['year'].value_counts().reset_index()
        yearly_counts.columns = ['year', 'publications']
        yearly_counts = yearly_counts.sort_values('year')
        
        # Calculate cumulative publications
        yearly_counts['cumulative'] = yearly_counts['publications'].cumsum()
        
        plot_data = yearly_counts
    
    # Create a subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for annual publications
    fig.add_trace(
        go.Bar(
            x=plot_data['year'],
            y=plot_data['publications'],
            name='Annual Publications',
            marker_color=COLORS[0],
            opacity=0.7,
        ),
        secondary_y=False,
    )
    
    # Add line chart for cumulative publications
    fig.add_trace(
        go.Scatter(
            x=plot_data['year'],
            y=plot_data['cumulative'],
            name='Cumulative Publications',
            mode='lines+markers',
            line=dict(color=COLORS[1], width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    
    # Add annotations for the peak year
    max_pub_year = plot_data.loc[plot_data['publications'].idxmax()]
    
    fig.add_annotation(
        x=max_pub_year['year'],
        y=max_pub_year['publications'],
        text=f"Peak: {max_pub_year['publications']} publications",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title='Publication Trend Over Time',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        barmode='group',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Update axes titles
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Annual Publications', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Publications', secondary_y=True)
    
    return fig

def plot_top_authors(data: pd.DataFrame, top_n: int = 15):
    """
    Create a horizontal bar chart showing authors with the most publications.
    
    Args:
        data: Processed bibliometric DataFrame
        top_n: Number of top authors to show
        
    Returns:
        Plotly figure
    """
    if data.empty or 'author_list' not in data.columns:
        # Generate sample data if no real data
        authors = [f'Author {chr(65+i)}' for i in range(top_n)]
        counts = [25, 22, 19, 18, 16, 15, 14, 12, 11, 10, 9, 8, 7, 6, 5]
        
        plot_data = pd.DataFrame({
            'author': authors,
            'count': counts
        })
    else:
        # Flatten the author lists and count occurrences
        all_authors = []
        for authors in data['author_list']:
            if isinstance(authors, list):
                all_authors.extend(authors)
        
        author_counts = Counter(all_authors)
        
        # Convert to DataFrame and sort
        plot_data = pd.DataFrame({
            'author': list(author_counts.keys()),
            'count': list(author_counts.values())
        })
        
        plot_data = plot_data.sort_values('count', ascending=False).head(top_n)
    
    # Sort ascending for horizontal bar chart
    plot_data = plot_data.sort_values('count', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        plot_data,
        x='count',
        y='author',
        orientation='h',
        labels={'count': 'Number of Publications', 'author': 'Author'},
        title=f'Top {top_n} Authors by Publication Count',
        color='count',
        color_continuous_scale='Viridis',
        text='count'
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        xaxis_title='Number of Publications',
        yaxis_title='Author',
    )
    
    return fig

def plot_keyword_cloud(data: pd.DataFrame):
    """
    Create a word cloud visualization of the most frequently used keywords.
    
    Args:
        data: Processed bibliometric DataFrame
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    if data.empty or 'keyword_list' not in data.columns:
        # Generate sample data if no real data
        keywords = {
            'machine learning': 45, 'artificial intelligence': 38, 'deep learning': 35,
            'neural networks': 30, 'data science': 28, 'big data': 25,
            'natural language processing': 22, 'computer vision': 20, 'algorithms': 18,
            'data mining': 16, 'pattern recognition': 15, 'image processing': 14,
            'information retrieval': 12, 'knowledge discovery': 10, 'sentiment analysis': 9
        }
    else:
        # Flatten the keyword lists and count occurrences
        all_keywords = []
        for keywords in data['keyword_list']:
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        # Remove stop words and short terms
        filtered_keywords = []
        for kw in all_keywords:
            if isinstance(kw, str) and len(kw) > 2:
                filtered_keywords.append(kw)
        
        keyword_counts = Counter(filtered_keywords)
        keywords = dict(keyword_counts.most_common(50))  # Top 50 keywords
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=500, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        min_font_size=8
    ).generate_from_frequencies(keywords)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Frequent Keywords', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    return fig

def plot_author_network(data: pd.DataFrame, min_collaborations: int = 2, max_nodes: int = 50):
    """
    Create a network visualization of author collaborations.
    
    Args:
        data: Processed bibliometric DataFrame
        min_collaborations: Minimum number of collaborations to include in the network
        max_nodes: Maximum number of nodes to display
        
    Returns:
        Plotly figure
    """
    if data.empty or 'author_list' not in data.columns:
        # Generate sample network data if no real data
        G = nx.random_geometric_graph(30, 0.25)
        
        # Add attributes to nodes
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['name'] = f'Author {chr(65+i%26)}'
            G.nodes[node]['publications'] = np.random.randint(5, 30)
        
        # Rename edges to collaborations
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 5)
    else:
        # Create a graph
        G = nx.Graph()
        
        # Count collaborations between authors
        collaborations = {}
        
        # For each paper, add edges between all co-authors
        for authors in data['author_list']:
            if isinstance(authors, list) and len(authors) > 1:
                # Generate all possible pairs of co-authors
                for author1, author2 in combinations(authors, 2):
                    if author1 == author2:
                        continue
                        
                    pair = tuple(sorted([author1, author2]))
                    collaborations[pair] = collaborations.get(pair, 0) + 1
        
        # Add nodes for each author
        author_counts = {}
        for authors in data['author_list']:
            if isinstance(authors, list):
                for author in authors:
                    author_counts[author] = author_counts.get(author, 0) + 1
        
        # Filter to top authors by publication count
        top_authors = dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:max_nodes])
        
        for author, count in top_authors.items():
            G.add_node(author, name=author, publications=count)
        
        # Add edges for collaborations that meet the minimum threshold
        for (author1, author2), count in collaborations.items():
            if count >= min_collaborations and author1 in top_authors and author2 in top_authors:
                G.add_edge(author1, author2, weight=count)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    
    # If graph is empty, create a sample graph
    if len(G.nodes()) == 0:
        G = nx.random_geometric_graph(15, 0.3)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['name'] = f'Author {chr(65+i%26)}'
            G.nodes[node]['publications'] = np.random.randint(5, 15)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 3)
    
    # Generate positions using a force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]].get('weight', 1))
    
    # Scale edge width by weight
    max_width = 5
    min_width = 1
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        range_weight = max(1, max_weight - min_weight)
        scaled_weights = [min_width + ((w - min_weight) / range_weight) * (max_width - min_width) for w in edge_weights]
    else:
        scaled_weights = [min_width]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(120, 120, 120, 0.5)'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces
    node_x = []
    node_y = []
    node_size = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Use publication count for node size
        publications = G.nodes[node].get('publications', 1)
        node_size.append(publications)
        
        # Create hover text
        connections = len(list(G.neighbors(node)))
        node_text.append(f"Author: {G.nodes[node].get('name', str(node))}<br>"
                         f"Publications: {publications}<br>"
                         f"Collaborations: {connections}")
        
        # Color by degree (number of collaborations)
        node_color.append(connections)
    
    # Scale node size
    min_size = 10
    max_size = 30
    if node_size:
        min_value = min(node_size)
        max_value = max(node_size)
        range_value = max(1, max_value - min_value)
        scaled_sizes = [min_size + ((s - min_value) / range_value) * (max_size - min_size) for s in node_size]
    else:
        scaled_sizes = [min_size]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=scaled_sizes,
            line=dict(width=1, color='white'),
            colorbar=dict(
                title='Number of Collaborations',
                thickness=15,
                xanchor='left',
                titleside='right'
            )
        ),
        text=node_text,
        name='Authors'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Author Collaboration Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        clickmode='event+select',
                        annotations=[
                            dict(
                                ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                                ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
                                axref='x', ayref='y',
                                x=(pos[edge[0]][0] + pos[edge[1]][0]) / 2 + 0.01,
                                y=(pos[edge[0]][1] + pos[edge[1]][1]) / 2 + 0.01,
                                xref='x', yref='y',
                                showarrow=False,
                                text=str(G[edge[0]][edge[1]].get('weight', '')),
                                font=dict(size=9),
                                bgcolor='rgba(255, 255, 255, 0.7)',
                                borderpad=2
                            )
                            for edge in G.edges() if G[edge[0]][edge[1]].get('weight', 1) > min_collaborations
                        ]
                    ))
    
    return fig

def plot_top_institutions(data: pd.DataFrame, top_n: int = 10):
    """
    Create a bar chart showing the institutions with the most publications.
    
    Args:
        data: Processed bibliometric DataFrame
        top_n: Number of top institutions to show
        
    Returns:
        Plotly figure
    """
    if data.empty or 'primary_institution' not in data.columns:
        # Generate sample data if no real data
        institutions = [
            'Stanford University', 'MIT', 'Harvard University', 'UC Berkeley',
            'University of Oxford', 'University of Cambridge', 'ETH Zurich',
            'Imperial College London', 'University of Toronto', 'Tsinghua University'
        ]
        counts = [45, 42, 38, 35, 32, 30, 28, 25, 22, 20]
        
        plot_data = pd.DataFrame({
            'institution': institutions,
            'count': counts
        })
    else:
        # Count occurrences of each institution
        institution_counts = data['primary_institution'].value_counts().reset_index()
        institution_counts.columns = ['institution', 'count']
        
        # Filter out None/NaN values
        institution_counts = institution_counts.dropna()
        
        # Get top N institutions
        plot_data = institution_counts.head(top_n)
    
    # Sort data for better visualization
    plot_data = plot_data.sort_values('count', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        plot_data,
        x='count',
        y='institution',
        orientation='h',
        labels={'count': 'Number of Publications', 'institution': 'Institution'},
        title=f'Top {top_n} Institutions by Publication Count',
        color='count',
        color_continuous_scale='Viridis',
        text='count'
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        xaxis_title='Number of Publications',
        yaxis_title='Institution',
    )
    
    return fig

def plot_countries(data: pd.DataFrame, top_n: int = 10):
    """
    Create a bar chart showing countries with the most publications.
    
    Args:
        data: Processed bibliometric DataFrame
        top_n: Number of top countries to show
        
    Returns:
        Plotly figure
    """
    if data.empty or 'primary_country' not in data.columns:
        # Generate sample data if no real data
        countries = [
            'United States', 'China', 'United Kingdom', 'Germany',
            'Canada', 'Australia', 'France', 'Japan', 'Italy', 'India'
        ]
        counts = [85, 65, 50, 42, 38, 35, 32, 30, 28, 25]
        
        plot_data = pd.DataFrame({
            'country': countries,
            'count': counts
        })
    else:
        # Count occurrences of each country
        country_counts = data['primary_country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        
        # Filter out None/NaN values
        country_counts = country_counts.dropna()
        
        # Get top N countries
        plot_data = country_counts.head(top_n)
    
    # Sort data for better visualization
    plot_data = plot_data.sort_values('count', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        plot_data,
        x='count',
        y='country',
        orientation='h',
        labels={'count': 'Number of Publications', 'country': 'Country'},
        title=f'Top {top_n} Countries by Publication Count',
        color='count',
        color_continuous_scale='Viridis',
        text='count'
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        xaxis_title='Number of Publications',
        yaxis_title='Country',
    )
    
    return fig

def plot_database_quality(data: pd.DataFrame):
    """
    Create a bar chart ranking databases by quality metrics.
    
    Args:
        data: Processed bibliometric DataFrame
        
    Returns:
        Plotly figure
    """
    if data.empty or 'source_db' not in data.columns or 'citation_count' not in data.columns:
        # Generate sample data if no real data
        databases = ['Scopus', 'Web of Science', 'IEEE Xplore', 'PubMed']
        avg_citations = [42.5, 38.2, 24.7, 18.9]
        avg_completeness = [0.92, 0.89, 0.78, 0.72]
        quality_score = [0.85, 0.80, 0.65, 0.55]
        
        plot_data = pd.DataFrame({
            'database': databases,
            'avg_citations': avg_citations,
            'avg_completeness': avg_completeness,
            'quality_score': quality_score
        })
    else:
        # Calculate quality metrics by database
        metrics = []
        
        for db, group in data.groupby('source_db'):
            metrics.append({
                'database': db,
                'avg_citations': group['citation_count'].mean(),
                'avg_completeness': group['completeness'].mean() if 'completeness' in group.columns else 0.5,
                'quality_score': group['quality_score'].mean() if 'quality_score' in group.columns else 0.5,
                'count': len(group)
            })
        
        plot_data = pd.DataFrame(metrics)
    
    # Sort by quality score
    plot_data = plot_data.sort_values('quality_score', ascending=True)
    
    # Create a subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for quality score
    fig.add_trace(
        go.Bar(
            x=plot_data['quality_score'],
            y=plot_data['database'],
            orientation='h',
            name='Quality Score',
            marker_color=COLORS[0],
            width=0.5,
            text=plot_data['quality_score'].apply(lambda x: f'{x:.2f}'),
            textposition='outside',
        ),
        secondary_y=False,
    )
    
    # Add scatter points for average citations
    fig.add_trace(
        go.Scatter(
            x=plot_data['avg_citations'],
            y=plot_data['database'],
            mode='markers',
            name='Avg. Citations',
            marker=dict(
                color=COLORS[1],
                size=12,
                symbol='diamond',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=plot_data['avg_citations'].apply(lambda x: f'{x:.1f}'),
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title='Database Quality Ranking',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder': 'total ascending'},
    )
    
    # Update axes titles
    fig.update_xaxes(title_text='Quality Score (0-1)', range=[0, max(1, plot_data['quality_score'].max() * 1.2)])
    fig.update_yaxes(title_text='Database', secondary_y=False)
    fig.update_yaxes(title_text='Average Citations', secondary_y=True, range=[0, max(1, plot_data['avg_citations'].max() * 1.5)])
    
    return fig