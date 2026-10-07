import re
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import BaseModel,Field

from transformers import pipeline

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

MODEL_DIR = Path(__file__).parent / "emotion_model"

emotion_model = pipeline(
    "text-classification",
    model=str(MODEL_DIR),   # local folder instead of the repo name
    top_k=None,
    device=-1,
)

SENTIMENT_MAP  = {
    # Positive
    "joy": "positive",

    # Neutral / ambiguous
    "surprise": "neutral",
    "neutral": "neutral",

    # Negative
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
}

INTENSIFIERS = {"very", "so", "really", "extremely", "totally", "absolutely", "literally"}
URGENT_WORDS = {"urgent", "asap", "immediately", "now", "emergency", "quickly", "hurry"}
SLANG = {"lol", "bro", "dude", "gonna", "wanna", "pls", "plz", "omg", "btw", "idk"}

class Tone(BaseModel):
    formality: str
    urgency: str
    is_task: bool = Field(default=False, description="identifies if the message is user ans to do any task or not")
    polite: bool


class Emotion_sentiment_intensity_Tone(BaseModel):
    emotion: str
    sentiment: str
    intensity: float
    tone: Tone

class Detect_emotion:
    def get_emotion(test: str):
        output = emotion_model(test)
        max_emoptions = max(output[0], key=lambda x: x['score'])
        return max_emoptions
    
    def get_sentiment(emotion: str):
        return SENTIMENT_MAP.get(emotion, "unknown")
    
    def get_intensity(text: str, emotion_conf: float) -> float:
        words = re.findall(r"[a-zA-Z']+", text)
        if not words:
            return 0.0
        caps = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
        exclaim = min(text.count("!"), 3) / 3
        repeats = 1.0 if re.search(r"(.)\1{2,}", text) else 0.0   # "sooooo"
        boosters = min(sum(w.lower() in INTENSIFIERS for w in words), 2) / 2

        score = (0.4 * emotion_conf + 0.2 * caps + 0.2 * exclaim
                + 0.1 * repeats + 0.1 * boosters)
        return round(min(score, 1.0), 2)
    
    def get_style(text: str) -> Tone:
        words = set(re.findall(r"[a-z']+", text))
        casual = bool(words & SLANG) or bool(re.search(r"[\U0001F300-\U0001FAFF]", text)) or text.islower()
        return {
            "formality": "casual" if casual else "formal",
            "urgency": "urgent" if (words & URGENT_WORDS) else "relaxed",
            "is_task": "?" in text,
            "polite": bool(words & {"please", "thanks", "thank", "kindly", "appreciate"}),
            } # pyright: ignore[reportReturnType]


def get_emotion_sentiment_intensity_tone(text: str) -> Emotion_sentiment_intensity_Tone:
    emotion_data = Detect_emotion.get_emotion(text) # type: ignore
    sentiment = Detect_emotion.get_sentiment(emotion = emotion_data['label'])
    intensity = Detect_emotion.get_intensity(text, emotion_data['score'])
    style = Detect_emotion.get_style(text)
    
    return {
        "emotion": emotion_data['label'],
        "sentiment": sentiment,
        "intensity": intensity,
        "tone": style
    } # pyright: ignore[reportReturnType]

class with_LLm():
    def get_emotion_sentiment_intensity_tone_with_LLM(text: str):
        llm_model = ChatGroq(
        model='openai/gpt-oss-20b'
        )
        stature_out = llm_model.with_structured_output(Emotion_sentiment_intensity_Tone).invoke(
            [SystemMessage(content="You are an expert in analyzing emotions, sentiments, intensity, and tone of text."),
            HumanMessage(content=f"Analyze the following text and provide the emotion, sentiment, intensity (0.0 to 1.0), and tone (formality, urgency, is_question, polite) in a structured format: '{text}'")]
        )
        return stature_out