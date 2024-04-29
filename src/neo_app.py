import numpy as np

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral

from spirals_umap_tsne import load_data, apply_tsne, apply_umap, apply_spiral, apply_neo
from spirals_umap_tsne import plot_mnist, manual_apply_neo, dict_manual_apply_neo, plot_embedding

from synthetic import generate_synthetic


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
    print(f"len(images) = {len(images)}")
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

    # for j in range(1, 10, 1):
    #     num_spirals = j
    #     for i in range(4, 8, 1):
    #         neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #         plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for j in range(1, 10, 1):
    #     num_spirals = j
    #     for i in range(4, 8, 1):
    #         neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #         plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    for i in range(4, 10, 1):
        num_points = i
        for j in range(0, 3, 1):
            num_spirals = np.power(num_points, j)
            neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=num_points)
            plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {num_points} |", neo_results, labels)


def synthetic_test():
    print('Begin synthetic_test')
    random_state = 42
    n_samples = 4000
    n_centers = 5
    n_features = 128

    # num_spirals = 10
    # num_points = 10

    data_dict, X, y, X_pca, pca = generate_synthetic(n_samples=n_samples,
                                                     n_centers=n_centers,
                                                     n_features=n_features,
                                                     random_state=random_state)

    # neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
    # plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
    #                'sklearn synthetic', neo_results, labels)

    for num_spirals in range(1, 10, 1):
        for num_points in range(1, 10, 1):
            neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
            plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
                           'sklearn synthetic', neo_results, labels)


if __name__ == '__main__':
    # main()
    # spiral_props_loop()
    synthetic_test()
