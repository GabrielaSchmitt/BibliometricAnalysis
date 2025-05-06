# 📚 Bibliometric Analysis Tool

![Project GIF](placeholder_gif.gif) 

## 🌐 Project Website
[[Bibliometric Analysis](https://biblioanalysis.streamlit.app/)] 
<br>

## 🎯 Objective
This tool provides a comprehensive bibliometric analysis of research publications by processing RIS files exported from academic databases. It helps researchers:
- Analyze publication trends over time
- Identify top authors, institutions, and countries
- Visualize collaboration networks
- Compare data quality across different databases
- Gain insights into research impact through citation analysis
<br>

## 🔍 Getting Started

### Prerequisites
1. **Data Collection**: Before using this tool, conduct your literature search on academic databases like:
   - [IEEE Xplore](https://ieeexplore.ieee.org/)
   - [Scopus](https://www.scopus.com/)
   - [Web of Science](https://www.webofscience.com/)
   - [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
   
2. **Export RIS Files**: 
   - After completing your search, export the results in RIS format
   - Most databases provide this option in their export/download menu
<br>

### Usage
1. Download required libraries
   ```
    pip install -r requirements.txt
   ```
2. Run the app

   ```
    streamlit run streamlit_app.py
   ```
3. Follow the in-app instructions:

   - Upload RIS files from different databases
   - Tag each file with its source database
   - Click "Process and Analyze" to generate visualizations
<br>

### 📊 Key Features

- **Multi-database Analysis:** Combine and compare results from different sources
- **Duplicate Removal:** Intelligent deduplication based on titles and DOIs
- **Interactive Visualizations:**
   - Publication trends over time
   - Author collaboration networks
   - Keyword clouds
   - Institutional and geographical analysis
- **Quality Assessment:** Compare database quality metrics
- **Data Export:** Download processed data for further analysis
<br>

### 📂 Project Structure
```
bibliometric-analysis-tool/
├── streamlit_app.py          # Main application file
├── modules/
│   ├── ris_parser.py         # RIS file parsing logic
│   ├── data_processor.py     # Data cleaning and processing
│   ├── visualizations.py     # Visualization generation
│   └── utils.py              # Helper functions
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

<br>

### 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

