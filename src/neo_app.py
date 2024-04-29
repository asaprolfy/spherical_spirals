import numpy as np

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral

from spirals_umap_tsne import load_data, apply_tsne, apply_umap, apply_spiral, apply_neo
from spirals_umap_tsne import plot_mnist


def main():
    print('Begin neo_app.py')

    images, labels = load_data()

    spiral_results = apply_spiral(images)
    plot_mnist('spiral', spiral_results, labels)

    neo_results = apply_neo(images)
    plot_mnist('neo', neo_results, labels)


if __name__ == '__main__':
    main()
