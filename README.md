# Narrative Consistency Auditor

A high-performance AI system that determines whether character backstories are consistent with 100k+ word novels using semantic retrieval, LLM reasoning, and machine learning arbitration.

## 🏗️ Architecture

The project follows a modular architecture with specialized components:

```
Dataset-20260110T122008Z-1-001/
├── src/
│   ├── processor.py    # Text cleaning and semantic chunking
│   ├── indexer.py      # Embedding generation and dual-query retrieval
│   └── reasoner.py     # LLM-based consistency analysis
├── main.py             # Orchestration and ML arbitration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment configuration template
└── Dataset/
    └── Books/          # Novel text files (Project Gutenberg format)
```

## 🚀 Key Features

### 1. **Intelligent Text Processing** ([src/processor.py](src/processor.py))
- Automatically strips Project Gutenberg headers/footers
- Semantic paragraph-aware chunking (~1200 tokens, 200 overlap)
- Preserves narrative causal links across chunk boundaries

### 2. **Dual-Query Retrieval** ([src/indexer.py](src/indexer.py))
- Uses `BAAI/bge-base-en-v1.5` sentence embeddings
- FAISS inner product search for high-performance retrieval
- Executes two complementary searches per query:
  - Raw backstory claim
  - Character-focused query (Name + claim)
- Automatic deduplication and top-k selection

### 3. **Structured LLM Reasoning** ([src/reasoner.py](src/reasoner.py))
- Powered by Google Gemini 1.5 Flash
- Forces structured JSON output with:
  - `analysis`: Detailed reasoning
  - `verdict`: CONSISTENT/CONTRADICT/UNCLEAR
  - `confidence`: 0.0-1.0 score
- Low-temperature generation for consistent results

### 4. **ML-Based Final Arbitration** ([main.py](main.py))
- Random Forest classifier combines:
  - Retrieval features (max/mean/min/std similarity)
  - LLM verdict and confidence
- Trained on labeled examples for robust predictions
- Fallback to LLM-only mode when untrained

## 📦 Installation

1. **Clone and navigate to the project:**
```bash
cd Dataset-20260110T122008Z-1-001
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google AI API key
# Get your key from: https://makersuite.google.com/app/apikey
```

4. **Add novels to Dataset/Books/:**
```bash
# Place your Project Gutenberg .txt files in Dataset/Books/
```

## 🎯 Usage

### Basic Usage - Verify a Backstory Claim

```python
from main import NarrativeConsistencyAuditor

# Initialize the auditor
auditor = NarrativeConsistencyAuditor()

# Load and index a novel
auditor.load_and_index_novel("Dataset/Books/The Count of Monte Cristo.txt")

# Verify a backstory claim
result = auditor.verify_backstory(
    character_name="Edmond Dantès",
    backstory_claim="was falsely imprisoned in the Château d'If"
)

print(f"Prediction: {result['prediction_label']}")
print(f"LLM Analysis: {result['llm_analysis']}")
```

### Advanced Usage - Train ML Arbitrator

```python
# Prepare training data CSV with columns:
# character_name, backstory_claim, label (1=consistent, 0=contradictory)

# Train the classifier
metrics = auditor.train_arbitrator(
    training_data_path="train.csv",
    test_size=0.2,
    save_model_path="models/arbitrator.pkl"
)

print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
```

### Running the Demo

```bash
python main.py
```

## 🔬 Technical Details

### Chunking Strategy
- Splits at paragraph boundaries (`\n\n`)
- Aggregates to ~1200 tokens per chunk
- 200-token overlap maintains context across boundaries
- Uses `tiktoken` (cl100k_base) for accurate token counting

### Retrieval Pipeline
1. Generate embeddings for all novel chunks
2. Build FAISS IndexFlatIP (inner product for normalized vectors)
3. For each query:
   - Embed query and character-focused variant
   - Search top-k*3 candidates from both
   - Deduplicate and rank by similarity
   - Return top-8 segments

### LLM Integration
- Model: `gemini-1.5-flash`
- Temperature: 0.1 (consistent reasoning)
- Prompt engineering enforces JSON schema
- Robust parsing with fallback handling

### Feature Engineering
```python
features = [
    max_similarity,      # Strongest evidence signal
    mean_similarity,     # Overall context relevance
    min_similarity,      # Weakest evidence
    std_similarity,      # Evidence consistency
    llm_verdict_encoded, # -1/0/1 for CONTRADICT/UNCLEAR/CONSISTENT
    llm_confidence       # 0.0-1.0 LLM certainty
]
```

## 📊 Performance Considerations

- **Novel Size**: Optimized for 100k+ word novels
- **Chunk Count**: ~800-1200 chunks for typical novel
- **Indexing Time**: ~30-60 seconds (CPU), ~10-20 seconds (GPU)
- **Query Time**: ~2-3 seconds per verification
- **Memory**: ~500MB for embeddings + model weights

## 🔧 Configuration Options

Environment variables in `.env`:

```bash
GOOGLE_API_KEY=your_key_here
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # Optional
GEMINI_MODEL=gemini-1.5-flash          # Optional
CHUNK_SIZE=1200                         # Optional
CHUNK_OVERLAP=200                       # Optional
TOP_K_RETRIEVAL=8                       # Optional
```

## 📝 Training Data Format

CSV file for `train_arbitrator()`:

```csv
character_name,backstory_claim,label
"Edmond Dantès","was a sailor who became wealthy","1"
"Edmond Dantès","was born in Paris","0"
"Mercedes","was engaged to Edmond","1"
```

Labels: `1` = Consistent, `0` = Contradictory

## 🎓 Example Workflow

```python
# 1. Initialize
auditor = NarrativeConsistencyAuditor()

# 2. Index novel
auditor.load_and_index_novel("Dataset/Books/In search of the castaways.txt")

# 3. Save index for reuse (optional)
auditor.indexer.save_index("novel.index", "novel.meta")

# 4. Load index later (skip re-indexing)
# auditor.indexer.load_index("novel.index", "novel.meta")

# 5. Verify multiple claims
claims = [
    ("Captain Grant", "was shipwrecked"),
    ("Lord Glenarvan", "owned a yacht"),
]

for character, claim in claims:
    result = auditor.verify_backstory(character, claim)
    print(f"{character}: {result['prediction_label']}")
```

## 🛠️ Extending the System

### Adding Pathway Integration

For real-time data orchestration, integrate Pathway:

```python
import pathway as pw

# Stream novel updates
novels = pw.io.fs.read("Dataset/Books/", mode="streaming")
# Process and index in real-time
# See Pathway documentation for details
```

### Custom Embedding Models

```python
auditor = NarrativeConsistencyAuditor()
auditor.indexer = DualQueryIndexer(model_name='sentence-transformers/all-mpnet-base-v2')
```

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **BAAI** for bge-base-en-v1.5 embeddings
- **Google** for Gemini LLM
- **Facebook Research** for FAISS
- **Project Gutenberg** for public domain texts
