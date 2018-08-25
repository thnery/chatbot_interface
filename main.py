#!/usr/bin/env python3

import csv
import click
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

global vectorizer
vectorizer = HashingVectorizer(n_features=3)

def load_data():
    texts = []
    with open('dialogo.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            texts.append(row[1])

    return texts

def train():
    data = load_data()
    vectorizer.fit_transform(data)

    vector = vectorizer.transform(data)

    result = np.array(vector.toarray())

    np.save('result', result)
    #print(np.load('result.npy'))

def transform_input(text):
    return vectorizer.transform([text])

@click.command()
@click.option('--text', prompt='Type a text', help='The text to be predicted')
def main(text):
    print(transform_input(text).toarray())
    vector = transform_input(text).toarray()

if __name__ == '__main__':
    train()
    main()
