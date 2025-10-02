# Mental Health Conversation Topic Modeling

A comprehensive topic modeling analysis of mental health counseling conversations using Latent Dirichlet Allocation (LDA) and various visualization techniques.

## Project Overview

This project analyzes mental health counseling conversations to identify key topics and themes discussed by individuals seeking mental health support. The analysis uses advanced natural language processing techniques to extract meaningful insights from textual data, helping to understand the most common concerns and themes in mental health conversations.

### Objectives

- **Topic Discovery**: Identify the main themes and topics in mental health conversations
- **Data Visualization**: Create comprehensive visualizations to understand topic distributions
- **Model Evaluation**: Assess topic model quality using coherence metrics
- **Interactive Analysis**: Provide interactive visualizations for deeper exploration

## Dataset

The project uses the **Mental Health Counseling Conversations** dataset from Hugging Face, which contains:

- **Source**: `Amod/mental_health_counseling_conversations` from Hugging Face Datasets
- **Format**: Tab-separated CSV with Context and Response columns
- **Size**: 6,295 conversation pairs
- **Unique Contexts**: 995 unique conversation contexts
- **Content**: Real mental health counseling conversations covering various psychological topics

### Sample Data
```
Context: "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here..."

Response: "If everyone thinks you're worthless, then maybe you need to find new people to hang out with..."
```

## Technical Implementation

### Core Technologies

- **Python 3.11** with virtual environment
- **Jupyter Notebook** for interactive analysis
- **Natural Language Processing**: NLTK, spaCy, Gensim
- **Machine Learning**: scikit-learn, Latent Dirichlet Allocation
- **Visualization**: Matplotlib, Seaborn, WordCloud, pyLDAvis, t-SNE
- **Data Processing**: Pandas, NumPy

### Key Libraries

```python
# Core NLP and ML
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Visualization
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud
from sklearn.manifold import TSNE

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
```

## Methodology

### 1. Data Preprocessing

- **Text Cleaning**: Convert to lowercase, remove special characters and numbers
- **Tokenization**: Split text into individual words
- **Stop Word Removal**: Remove common English stop words and domain-specific terms
- **Lemmatization**: Reduce words to their base forms using WordNetLemmatizer
- **Filtering**: Remove words shorter than 2 characters

### 2. Topic Modeling Approaches

#### Approach 1: Gensim LDA (Primary Method)
- **Model**: Gensim LdaModel with auto-tuned parameters
- **Topics**: 10 topics (determined by coherence analysis)
- **Parameters**:
  - `chunksize`: 2000
  - `passes`: 10
  - `iterations`: 100
  - `alpha`: 'auto'
  - `eta`: 'auto'

#### Approach 2: Scikit-learn LDA (Secondary Method)
- **Model**: LatentDirichletAllocation
- **Vectorization**: TF-IDF with max_features=1000
- **Topics**: 10 topics
- **Parameters**:
  - `n_components`: 10
  - `random_state`: 42
  - `max_iter`: 20

### 3. Model Evaluation

- **Coherence Analysis**: C_v coherence metric to determine optimal number of topics
- **Perplexity**: Model perplexity for quality assessment
- **Visual Validation**: Multiple visualization techniques for topic interpretation

## Results and Findings

### Discovered Topics

The analysis identified **10 distinct topics** in mental health conversations:

1. **Topic 1 - Academic & Life Struggles**
   - Keywords: thinking, school, someone, something, tell, men, life, back, stop, year

2. **Topic 2 - Emotional Support & Decision Making**
   - Keywords: need, life, stop, cry, decision, tell, everything, would, friend, empty

3. **Topic 3 - Social Relationships & Trust**
   - Keywords: people, normal, trust, work, want, love, anything, act, friend, wife

4. **Topic 4 - Anger & Problem Resolution**
   - Keywords: feeling, wrong, anger, better, problem, make, girlfriend, fix, room, nothing

5. **Topic 5 - Intimate Relationships**
   - Keywords: love, sex, told, said, much, week, still, boyfriend, guy, day

6. **Topic 6 - Emotional Instability & Diagnosis**
   - Keywords: girl, easily, split, married, anger, different, lost, got, emotional, diagnosed

7. **Topic 7 - Family Dynamics**
   - Keywords: dad, thing, mom, friend, live, good, kid, together, get, boyfriend

8. **Topic 8 - Substance Abuse & Social Issues**
   - Keywords: asked, people, drug, social, boyfriend, girl, wife, abuse, make, could

9. **Topic 9 - Childhood & Family Trauma**
   - Keywords: child, mother, saying, day, voice, people, thing, would, dog, sister

10. **Topic 10 - Depression & Self-Discovery**
    - Keywords: find, could, someone, fear, start, understand, depressed, even, answer, try

### Key Insights

- **Relationship Issues**: Topics 3, 5, and 7 indicate significant focus on interpersonal relationships
- **Mental Health Conditions**: Topics 6 and 10 specifically relate to diagnosed conditions and depression
- **Family Dynamics**: Topics 7 and 9 highlight family-related concerns and childhood experiences
- **Substance Abuse**: Topic 8 reveals substance abuse as a recurring theme
- **Emotional Regulation**: Topics 2 and 4 focus on emotional management and problem-solving

## Visualizations

The project includes comprehensive visualizations:

### 1. **Word Count Distribution**
- Histogram showing document length distribution
- Analysis of text complexity across conversations

### 2. **Topic-Specific Word Clouds**
- Visual representation of top keywords for each topic
- Color-coded word clouds showing topic themes

### 3. **t-SNE Clustering**
- 2D projection of document-topic relationships
- Cluster visualization showing topic separation

### 4. **Interactive pyLDAvis**
- Interactive topic visualization
- Topic-term and document-topic exploration
- Saved as `lda_visualization.html`

### 5. **Coherence Analysis**
- Line plot showing coherence scores across different topic numbers
- Optimal topic number identification

## Getting Started

### Prerequisites

- Python 3.11+
- Virtual environment support
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mentalHealthTM
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv mentalHealthTMenv
   source mentalHealthTMenv/bin/activate  # On Windows: mentalHealthTMenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Usage

1. **Run the main analysis notebook**
   ```bash
   jupyter notebook MentalHealthConversation_topicModelling.ipynb
   ```

2. **Explore data preprocessing**
   ```bash
   jupyter notebook dataExplore.ipynb
   ```

3. **View interactive visualizations**
   - Open `lda_visualization.html` in your browser
   - Explore topic relationships and term distributions

## Project Structure

```
mentalHealthTM/
├── MentalHealthConversation_topicModelling.ipynb  # Main analysis notebook
├── dataExplore.ipynb                              # Data exploration and preprocessing
├── mental_health_counseling_conversations.csv     # Processed dataset
├── lda_visualization.html                         # Interactive LDA visualization
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project documentation
└── mentalHealthTMenv/                            # Virtual environment
```

## Configuration

### Model Parameters

The project uses configurable parameters for different aspects:

```python
# Topic Modeling Parameters
NUM_TOPICS = 10
chunksize = 2000
passes = 6
iterations = 100

# Text Processing Parameters
max_features = 1000
min_df = 2
max_df = 0.95
```

### Custom Stop Words

The analysis includes domain-specific stop words to improve topic quality:

```python
custom_stop_words = {
    'like', 'feel', 'want', 'know', 'time', 'never', 'really', 'think', 
    'therapy', 'therapist', 'counselor', 'counseling', 'doctor',
    'mental', 'health', 'anxiety', 'depression', 'disorder', 'stress'
}
```

## Performance Metrics

### Model Quality Indicators

- **Coherence Score (C_v)**: Used to determine optimal number of topics
- **Perplexity**: Model quality assessment
- **Convergence**: Document convergence rates during training

### Dataset Statistics

- **Total Conversations**: 6,295
- **Unique Contexts**: 995
- **Vocabulary Size**: 88 unique tokens (after filtering)
- **Average Document Length**: Variable (analyzed in word count distribution)

## Contributing

Contributions are welcome! Please consider:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

### Areas for Enhancement

- **Advanced Preprocessing**: Implement more sophisticated text cleaning
- **Topic Validation**: Add human evaluation of topic quality
- **Temporal Analysis**: Analyze topic trends over time
- **Sentiment Analysis**: Combine with sentiment analysis for deeper insights
- **Multi-language Support**: Extend to support multiple languages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For providing the mental health counseling conversations dataset
- **Amod**: Dataset creator for the mental health counseling conversations
- **Open Source Community**: For the excellent NLP and visualization libraries

## Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check this README and notebook comments for detailed explanations

## Research Applications

This project has potential applications in:

- **Clinical Psychology**: Understanding common patient concerns
- **Mental Health Research**: Identifying trends in mental health conversations
- **Therapeutic Training**: Training materials for counselors and therapists
- **Public Health**: Informing mental health awareness campaigns
- **Natural Language Processing**: Benchmark for topic modeling in healthcare text

---

*This project demonstrates the power of topic modeling in understanding mental health conversations and provides a foundation for further research in healthcare NLP.*