import os
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


DATA_PATH = "data/mal_anime.csv"
TOP_N = 10


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[
        [
            "title", "description", "Genres", "Themes",
            "Type", "Rating", "Score",
            "Popularity", "Members", "Favorites"
        ]
    ]

    df.fillna("", inplace=True)

    for col in ["Score", "Popularity", "Members", "Favorites"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"Dataset loaded ({len(df)} rows)")
    return df


# ---------------------------
# TF-IDF RECOMMENDER MODEL
# ---------------------------
def train_recommender(df):
    print("Training TF-IDF recommendation model...")
    os.makedirs("models", exist_ok=True)

    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["Genres"] + " " +
        df["Themes"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")

    print("TF-IDF model trained and saved")
    return tfidf_matrix


# ---------------------------
# POPULARITY PREDICTION MODEL
# ---------------------------
def train_popularity_model(df):
    print("Training popularity prediction model...")
    os.makedirs("models", exist_ok=True)

    threshold = df["Favorites"].median()
    df["popular"] = (df["Favorites"] >= threshold).astype(int)

    X = df[["Score", "Members"]]
    y = df["popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print("\nPopularity model classification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Popular", "Popular"]
    ))

    print(f"Overall accuracy: {accuracy:.2f}")

    joblib.dump(model, "models/popularity_model.pkl")
    print("Popularity model saved")

    return model, accuracy


# ---------------------------
# SIMILAR ANIME
# ---------------------------
def recommend_similar_anime(title, df, tfidf_matrix, pop_model, age_filter=None):
    title = title.lower()

    if title not in df["title"].str.lower().values:
        print("Anime not found.")
        return None

    idx = df[df["title"].str.lower() == title].index[0]

    similarity = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    recs = df.copy()
    recs["similarity"] = similarity
    recs = recs[recs.index != idx]

    if age_filter:
        recs = recs[recs["Rating"].str.contains(age_filter, case=False)]

    probs = pop_model.predict_proba(
        recs[["Score", "Members"]]
    )[:, 1]

    recs["popularity_prob"] = probs

    recs = recs.sort_values(
        by=["similarity", "popularity_prob", "Score"],
        ascending=False
    )

    return recs.head(TOP_N)


# ---------------------------
# GENRE BASED
# ---------------------------
def recommend_by_genre(df, genre):
    subset = df[df["Genres"].str.contains(genre, case=False, na=False)]

    if subset.empty:
        print("No anime found for this genre.")
        return None

    return subset.sort_values(
        by=["Score", "Favorites"],
        ascending=False
    ).head(TOP_N)


# ---------------------------
# SEARCH
# ---------------------------
def search_titles(df, keyword):
    results = df[df["title"].str.contains(keyword, case=False, na=False)]
    return results.head(15) if not results.empty else None


# ---------------------------
# MAIN PROGRAM
# ---------------------------
def main():
    print("=" * 70)
    print(" Intelligent Anime Recommendation System (ML Integrated)")
    print("=" * 70)

    df = load_data()
    tfidf_matrix = train_recommender(df)
    pop_model, pop_acc = train_popularity_model(df)

    print("\nMODEL PERFORMANCE SUMMARY")
    print("-" * 40)
    print("TF-IDF Recommendation Accuracy : N/A (Unsupervised)")
    print(f"Popularity Prediction Accuracy : {pop_acc:.2f}")
    print("-" * 40)

    while True:
        print("\nChoose an option:")
        print("[1] Recommend anime similar to a given title")
        print("[2] Recommend top anime by genre")
        print("[3] Search for titles containing a keyword")
        print("[0] Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Goodbye 👋")
            break

        elif choice == "1":
            title = input("Enter anime title: ")
            age = input("Age rating filter (PG-13, R, or Enter to skip): ")

            results = recommend_similar_anime(
                title, df, tfidf_matrix, pop_model,
                age_filter=age if age else None
            )

            if results is not None:
                for i, row in results.iterrows():
                    print(f"\n{i+1}. {row['title']}")
                    print(f"Genres     : {row['Genres']}")
                    print(f"Type       : {row['Type']}")
                    print(f"Rating     : {row['Rating']}")
                    print(f"Score      : {row['Score']}")
                    print(f"Favorites  : {row['Favorites']}")
                    print(f"Popularity : {row['popularity_prob']*100:.1f}%")

        elif choice == "2":
            genre = input("Enter a genre: ")
            results = recommend_by_genre(df, genre)

            if results is not None:
                for i, row in results.iterrows():
                    print(f"\n{i+1}. {row['title']}")
                    print(f"Genres     : {row['Genres']}")
                    print(f"Type       : {row['Type']}")
                    print(f"Score      : {row['Score']}")
                    print(f"Members    : {row['Members']}")

        elif choice == "3":
            keyword = input("Enter keyword: ")
            results = search_titles(df, keyword)

            if results is not None:
                for _, row in results.iterrows():
                    print(f"- {row['title']} (Score: {row['Score']})")
            else:
                print("No results found.")

        else:
            print("Invalid choice ❌")


if __name__ == "__main__":
    main()
