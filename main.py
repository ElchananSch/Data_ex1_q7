import pandas as pd
import numpy as np


def preprocess_text(text):
    # Convert text to lowercase and split into words
    return text.lower().split()


def compute_tf(description, word_list):
    word_count = {}
    for word in description:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    tf = {}
    description_length = len(description)
    for word in word_list:
        tf[word] = word_count.get(word, 0) / description_length
    return tf


def compute_df(descriptions, word_list):
    df = {word: 0 for word in word_list}
    for description in descriptions:
        unique_words = set(description)
        for word in word_list:
            if word in unique_words:
                df[word] += 1
    return df


def compute_idf(df, total_documents):
    idf = {}
    for word, count in df.items():
        idf[word] = np.log((total_documents + 1) / (count + 1)) + 1
    return idf


def tfidf(df, word_list):
    # Read the dataset

    # Words to calculate TF-IDF for
    words = ["annual", "music", "festival", "soul", "jazz", "belgium", "hungary", "israel", "rock", "dance", "desert",
             "electronic", "arts"]

    # Preprocess descriptions
    df['Description'] = df['Description'].apply(preprocess_text)

    # Calculate Document Frequency (DF)
    descriptions = df['Description'].tolist()
    df_word = compute_df(descriptions, words)

    # Calculate Inverse Document Frequency (IDF)
    total_documents = len(descriptions)
    idf_word = compute_idf(df_word, total_documents)

    # Calculate TF and TF-IDF for each description
    tfidf_values = []
    for description in descriptions:
        tf = compute_tf(description, words)
        tfidf = {word: tf[word] * idf_word[word] for word in words}
        tfidf_values.append(tfidf)

    # Create DataFrame for the TF-IDF values
    tfidf_df = pd.DataFrame(tfidf_values)
    tfidf_df.index = df.index

    return tfidf_df




def main():
    df = pd.read_csv('music_festivals.csv')

    tfidf_df = tfidf()
    print(tfidf_df)












if __name__ == "__main__":
    main()
