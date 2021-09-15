import os
import re
import argparse
import pickle

import pandas as pd
import torch

from builder import PandasGraphBuilder


def movielens_graph_building(args):
    directory = args.directory

    users = []
    with open(os.path.join(directory, 'users.dat'), encoding='latin1') as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split('::')
            users.append({
                'user_id': int(id_),
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_,
            })
    users = pd.DataFrame(users).astype('category')

    movies = []
    with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {'movie_id': int(id_), 'title': title, 'year': year}
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

    distinct_users_in_ratings = ratings['user_id'].unique()
    distinct_movies_in_ratings = ratings['movie_id'].unique()
    users = users[users['user_id'].isin(distinct_users_in_ratings)]
    movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]

    genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
    movies_categorical = movies.drop('title', axis=1)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
    graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')
    graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')
    g = graph_builder.build()

    g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)
    g.edges['watched'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['watched-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    g = movielens_graph_building(args)

    # dump dataset
    dataset = {
        'train-graph': g,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'movie',
        'user-to-item-type': 'watched',
        'item-to-user-type': 'watched-by',
        'timestamp-edge-column': 'timestamp'}
    output_path = args.output_path
    output_path = os.path.join(output_path, 'graph_data.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
