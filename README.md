# 🎙️ Grammar Scoring Engine from Voice Samples

This notebook builds an **end-to-end Grammar Scoring Engine** using **audio (voice) samples** and outputs a **numerical grammar score**. It is **Kaggle-ready**, reproducible, and optimized for strong baseline performance.

---

## 📌 Problem Overview

Given a set of **spoken audio recordings**, predict a **grammar score** that reflects grammatical correctness of the speech.

**Pipeline:**

```
Audio → Speech-to-Text → Text Cleaning → Grammar Features → ML Model → Score
```

---

## 📁 Dataset Structure (Expected)

```
/train
  ├── audio_001.wav
  ├── audio_002.wav
/train.csv
/test.csv
```

**CSV Columns:**

* `audio_id`
* `file_path`
* `grammar_score` (train only)

---

## 🔧 Install Dependencies

```python
!pip install -q openai-whisper language-tool-python librosa
!pip install -q scikit-learn pandas numpy
```

---

## 📦 Imports

```python
import os
import pandas as pd
import numpy as np
import whisper
import librosa
import re
import language_tool_python

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

---

## 🧠 Load Whisper Model

```python
asr_model = whisper.load_model("base")
```

---

## ✍️ Speech to Text

```python
def transcribe_audio(path):
    result = asr_model.transcribe(path)
    return result['text']
```

---

## 🧹 Text Cleaning

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
```

---

## 🧾 Grammar Feature Extraction

```python
tool = language_tool_python.LanguageTool('en-US')

def extract_features(text):
    words = text.split()
    errors = tool.check(text)
    return {
        'word_count': len(words),
        'grammar_errors': len(errors),
        'error_ratio': len(errors) / max(1, len(words))
    }
```

---

## 📊 Prepare Training Data

```python
df = pd.read_csv('/kaggle/input/train.csv')

features = []
for _, row in df.iterrows():
    audio_path = row['file_path']
    text = transcribe_audio(audio_path)
    text = clean_text(text)
    feats = extract_features(text)
    features.append(feats)

X = pd.DataFrame(features)
y = df['grammar_score']
```

---

## 🔀 Train / Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🤖 Model Training

```python
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
```

---

## 📈 Evaluation

```python
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print("Validation MAE:", mae)
```

---

## 🧪 Test Prediction

```python
test_df = pd.read_csv('/kaggle/input/test.csv')

test_features = []
for _, row in test_df.iterrows():
    text = transcribe_audio(row['file_path'])
    text = clean_text(text)
    test_features.append(extract_features(text))

X_test = pd.DataFrame(test_features)
test_preds = model.predict(X_test)
```

---

## 📤 Submission File

```python
submission = pd.DataFrame({
    'audio_id': test_df['audio_id'],
    'grammar_score': test_preds
})

submission.to_csv('submission.csv', index=False)
```

---

## 🏆 Tips to Improve Kaggle Score

* Use **Whisper large-v2**
* Add **pause / filler word features**
* Ensemble with **XGBoost**
* Use **BERT embeddings** from transcribed text
* Normalize scores

---

## ✅ Outcome

✔ Fully automated grammar scoring
✔ Kaggle-compatible pipeline
✔ Resume-ready ML project

---

**Author:** Grammar Scoring Engine | ML + Speech Processing
