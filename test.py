
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text to summarize
text = """The Hugging Face library provides easy-to-use NLP models, including ones for summarization.
          This allows users to quickly generate summaries of long documents with minimal effort."""

# Summarize the text
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)

# Print the summary
print(summary[0]['summary_text'])
