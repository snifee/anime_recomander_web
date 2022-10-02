import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
anime_synopsis = pd.read_csv('anime_with_synopsis.csv')

anime_synopsis['sypnopsis'] = anime_synopsis['sypnopsis'].fillna('')


# tfidf = TfidfVectorizer(analyzer='word')
# tfidf_matrix = tfidf.fit_transform(anime_synopsis['sypnopsis'])
# tfidf_matrix.todense()

# similarity = cosine_similarity(tfidf_matrix) 

# similarity_df = pd.DataFrame(similarity, index=anime_synopsis['Name'], columns=anime_synopsis['Name'])

# similarity_df.to_excel(index=False)

def anime_recommendations(anime, similarity,items=anime_synopsis[['Name', 'sypnopsis','Genres']], k=5):

    # similarity_data=pd.read_pickle("./dummy.pkl")

    similarity_data = similarity

    index = similarity_data.loc[:,anime].to_numpy().argpartition(
        range(-1, -k, -1))
    
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    closest = closest.drop(anime, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)["Name"].values.tolist()