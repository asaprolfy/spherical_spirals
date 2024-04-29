import numpy as np

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral

from spirals_umap_tsne import load_data, apply_tsne, apply_umap, apply_spiral, apply_neo
from spirals_umap_tsne import plot_mnist, manual_apply_neo


def main():
    print('Begin neo_app.py')

    images, labels = load_data()

    tsne_results = apply_tsne(images)
    plot_mnist('tsne', tsne_results, labels)

    umap_results = apply_umap(images)
    plot_mnist('umap', umap_results, labels)

    spiral_results = apply_spiral(images)
    plot_mnist('spiral', spiral_results, labels)

    neo_results = apply_neo(images)
    plot_mnist('neo', neo_results, labels)


def spiral_props_loop():
    print('Begin spiral_props_loop')

    images, labels = load_data()
    print(f"shape: {images[0].shape}")
    print(f"len(vector): {len(images[0])}")

    num_spirals = 10

    # for i in range(500, 10000, 100):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for i in range(10, 200, 10):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for i in range(1, 10, 1):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    for j in range(1, 10, 1):
        num_spirals = j
        for i in range(4, 8, 1):
            neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
            plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)


if __name__ == '__main__':
    # main()
    spiral_props_loop()
