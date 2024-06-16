import re
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def unique_description(df_col):
    descriptions = df_col.apply(preprocess_text)

    # Initialize an empty set to store unique words
    unique_words = set()

    # Iterate through each inner list and add each word to the set
    for description in descriptions:
        for word in description:
            unique_words.add(word)

    return list(unique_words)


def preprocess_text(text):
    # Remove non-alphanumeric characters (including punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
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
    for word in word_list:
        tf[word] = word_count.get(word, 0)
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
        idf[word] = np.log(total_documents / count)
    return idf


def tfidf(df, words):
    # Preprocess descriptions
    df['Description'] = df['Description'].apply(preprocess_text)

    # DF
    descriptions = df['Description'].tolist()
    df_word = compute_df(descriptions, words)

    # IDF
    total_documents = len(descriptions)
    idf_word = compute_idf(df_word, total_documents)

    # TF and TF-ID
    tfidf_values = []
    for description in descriptions:
        tf = compute_tf(description, words)
        tfidf_t = {word: tf[word] * idf_word[word] for word in words}
        tfidf_values.append(tfidf_t)

    # Create DataFrame
    tfidf_df = pd.DataFrame(tfidf_values)
    tfidf_df.index = df.index

    return tfidf_df


def main():
    data = pd.read_csv('music_festivals.csv')
    # Words to calculate TF-IDF for

    # section 7.a.1
    words = ["annual", "music", "festival", "soul", "jazz", "belgium", "hungary", "israel", "rock", "dance", "desert",
             "electronic", "arts"]
    tfidf_df = tfidf(data, words)
    # Add the "Music Festival" column to tfidf_df
    tfidf_df['Music Festival'] = data['Music Festival']

    # Reorder the columns to make "Music Festival" the first column
    columns = ['Music Festival'] + [col for col in tfidf_df if col != 'Music Festival']
    tfidf_df = tfidf_df[columns]
    # screenshot of the df is in ex1_q7_answers
    tfidf_df.to_csv('tfidf_words.csv', index=False)

    # section 7.a.2 is manually
    pass

    # section 7.b is manually
    pass

    # section 7.c

    data = pd.read_csv('music_festivals.csv')

    # Extract relevant columns
    data['Year'] = data['Year'] - data['Year'].min()  # pre-process for year (more meaningful)
    numerical_columns = ['Year', 'Number of Participants', 'Ticket Price (in USD)', 'Number of Stages',
                         'Number of Music Genres']
    textual_columns = ['Description']

    # Normalize numerical data
    numerical_data = data[numerical_columns]
    numerical_data = (numerical_data - numerical_data.mean()) / numerical_data.std()

    # Process textual data
    unique_description_words = unique_description(data[textual_columns[0]])
    tfidf_all_words = tfidf(data, unique_description_words)

    # Compute pairwise distances for each data type
    numerical_distance = pairwise_distances(numerical_data, metric='euclidean')
    textual_distance = pairwise_distances(tfidf_all_words, metric='cosine')

    # combine distances
    total_distance = numerical_distance + textual_distance

    # Convert to DataFrame
    distance_df = pd.DataFrame(total_distance, index=data['Music Festival'], columns=data['Music Festival'])

    print(distance_df)


if __name__ == "__main__":
    main()
