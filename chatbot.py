import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

# Here you input a csv with a Question column and Answer column capitalized that way
dataframe = pd.read_csv('/content/frequently_asked_questions.csv')
dataframe.dropna(inplace=True) # This will drop nan values

vectorizer.fit(np.concatenate((dataframe.Question, dataframe.Answer)))
vectorized_questions = vectorizer.transform(dataframe.Question)
print(vectorizer_questions)

while True:
  user_input = input()
  vectorized_user_input = vectorizer.transform([user_input])
  similarities = cosine_similarity(vectorized_user_input, vectorized_questions)
  
  closest_question = np.argmax(similarities, axis=1)

  answer = dataframe.Answer.iloc[closest_question].values[0]
  print(answer)
  break # Break to stop the loop or keep going or set an escape input whatever floats your boat

