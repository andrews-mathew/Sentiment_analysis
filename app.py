import gradio as gr
from textblob import TextBlob

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

description_text = (
    "Enter a sentence, and the model will predict if it's POSITIVE, "
    "NEGATIVE, or NEUTRAL."
)

# Gradio Interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis API",
    description=description_text,
)

if __name__ == "__main__":
    interface.launch()
