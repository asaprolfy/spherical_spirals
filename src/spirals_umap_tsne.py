import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import plotly.express as px

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral

def load_data():
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    images = np.array(fashion_mnist.data, dtype=np.float32)
    labels = np.array(fashion_mnist.target, dtype=int)
    np.random.seed(42)
    indices = np.random.choice(len(images), 10000, replace=False)
    return images[indices], labels[indices]


def apply_tsne(data):
    try:
        tsne = TSNE(n_components=3, random_state=42)
        return tsne.fit_transform(data)
    except Exception as e:
        print(f"An error occurred while applying t-SNE: {e}")
        return None


def apply_umap(data):
    try:
        reducer = umap.UMAP(n_components=3, random_state=42)
        return reducer.fit_transform(data)
    except Exception as e:
        print(f"An error occurred while applying UMAP: {e}")
        return None


def apply_spiral(data):
    try:
        centroids = []
        for vector in data:
            spirals = SphericalSpiral(num_select=len(vector), num_points=1000)
            spirals.adjust_points(vector)
            centroids.append(spirals.find_centroid())
            spirals = None
        return np.array(centroids)
    except Exception as e:
        print(f"An error occurred while applying spherical spirals: {e}")
        return None


def apply_neo(data):
    try:
        centroids = []
        for vector in data:
            spirals = NeoSpiral(num_select=len(vector), num_points=1000)
            spirals.adjust_points(vector)
            centroids.append(spirals.find_centroid())
            spirals = None
        return np.array(centroids)
    except Exception as e:
        print(f"An error occurred while applying spherical spirals: {e}")
        return None


if __name__ == '__main__':
    print('Error: not a main file')
