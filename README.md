# Discrete sentiment

## Description
Project for sentiment analysis with ability to measure impact value of each word of input text.

## Run
```python3
from main import predict_sentiment

result = predict_sentiment("Текст для анализа сентимента")


Input: raw text string
Output: {"sentiment": [positive_class_probability,
                       neutral_class_probability,
                       negative_class_probability],
         "impact": [{"word": word,
                     "score": word_absolute_score,
                     "probability": word_impact_value}, ...]}
```
