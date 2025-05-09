import pandas as pd

# Load the CSV file
csv_path = "C:\\Users\\gabriela\\Downloads\\RIS EXPORTS\\EBSCO-Metadata-09_05_2025.csv"
df = pd.read_csv(csv_path)

# Define RIS tag mapping
ris_mapping = {
    'title': 'TI',
    'abstract': 'AB',
    'contributors': 'AU',
    'publicationDate': 'PY',
    'doi': 'DO',
    'plink': 'UR',
    'volume': 'VL',
    'issue': 'IS',
    'pageStart': 'SP',
    'pageEnd': 'EP',
    'publisher': 'PU',
    'language': 'LA',
    'notes': 'N1',
    'pubTypes': 'PT',
    'issns': 'SN',
}

# Function to determine RIS entry type
def get_ris_type(pub_type):
    if pd.isna(pub_type):
        return "GEN"
    pub_type = pub_type.lower()
    if "journal" in pub_type:
        return "JOUR"
    elif "conference" in pub_type:
        return "CPAPER"
    elif "book" in pub_type:
        return "BOOK"
    elif "newspaper" in pub_type:
        return "NEWS"
    else:
        return "GEN"

# Generate RIS entries
ris_entries = []
for _, row in df.iterrows():
    entry = []

    ris_type = get_ris_type(row.get("pubTypes"))
    entry.append(f"TY  - {ris_type}")

    authors = row.get("contributors")
    if pd.notna(authors):
        for author in authors.split(" ; "):
            entry.append(f"AU  - {author.strip()}")

    for col, tag in ris_mapping.items():
        value = row.get(col)
        if pd.notna(value):
            entry.append(f"{tag}  - {value}")

    entry.append("ER  - ")
    ris_entries.append("\n".join(entry))

output_path = "output_from_csv.ris"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(ris_entries))

print(f"RIS file successfully saved to {output_path}")