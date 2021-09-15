import os
import re
import argparse
import pickle

import pandas as pd
import torch

from builder import PandasGraphBuilder


def movielens_graph_building(args):
    directory = args.directory

    movies = []
    with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {'movie_id': int(id_), 'title': title, 'year': year, 'genre': genres.split("|")}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({'year': 'category'})

    ratings = []
    with open(os.path.join(directory, 'ratings.dat'), encoding='latin1') as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp,
            })
    ratings = pd.DataFrame(ratings)

    merged_ratings = pd.merge(ratings, movies, on=['movie_id'])
    merged_ratings = merged_ratings[['movie_id', 'rating', 'genre']]
    merged_ratings = merged_ratings.explode('genre')
    genres = pd.DataFrame(merged_ratings['genre'].unique()).reset_index()
    genres.columns = ['genre_id', 'genre']
    merged_ratings = pd.merge(merged_ratings, genres, on='genre')
    distinct_movies_in_ratings = merged_ratings['movie_id'].unique()
    movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]
    genres = pd.DataFrame(genres).astype({'genre_id': 'category'})

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(genres, 'genre_id', 'genre')
    graph_builder.add_entities(movies, 'movie_id', 'movie')
    graph_builder.add_binary_relations(merged_ratings, 'genre_id', 'movie_id', 'define')
    graph_builder.add_binary_relations(merged_ratings, 'movie_id', 'genre_id', 'define-by')
    g = graph_builder.build()
    
    g.nodes['genre'].data['id'] = torch.LongTensor(genres['genre_id'].cat.codes.values)
    movies = pd.DataFrame(movies).astype({'year': 'category'})
    genre_columns = movies.columns.drop(['movie_id', 'title', 'year', 'genre'])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
    g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)
    g.edges['define'].data['rating'] = torch.LongTensor(merged_ratings['rating'].values)
    g.edges['define-by'].data['rating'] = torch.LongTensor(merged_ratings['rating'].values)

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    # preprocess for graph building
    g = movielens_graph_building(args)

    # Dump the graph and the datasets
    dataset = {
        'train-graph': g,
        'context-type': 'genre',
        'item-type': 'movie',
        'context-to-item-type': 'define',
        'item-to-context-type': 'define-by'}

    output_path = args.output_path
    output_path = os.path.join(output_path, 'graph_data.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

# python context_query_preprocessor.py --directory=../dataset/movieLens --output_path=./