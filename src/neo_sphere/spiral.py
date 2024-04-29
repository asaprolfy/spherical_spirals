import numpy as np
import open3d as o3d


class NeoSpiral(object):
    def __init__(self, vector, manual=False, num_spirals=10, num_points=5000):
        self.vector = vector
        if manual:
            self.num_spirals = num_spirals
            self.num_points = num_points
        else:
            self.num_spirals = 10
            self.num_points = 5000
        self.num_select = len(vector)
        ####################################
        # generate_spaced_spherical_spiral()
        theta = np.linspace(0, self.num_spirals * 2 * np.pi, self.num_points)
        z = np.linspace(-1, 1, self.num_points)
        radius = np.sqrt(1 - z ** 2)
        self.x = radius * np.cos(theta)
        self.y = radius * np.sin(theta)
        self.z = z
        ####################################
        # select_evenly_spaced_points()
        indices = np.linspace(0, len(self.x) - 1, self.num_select, dtype=int)
        self.x_selected, self.y_selected, self.z_selected = self.x[indices], self.y[indices], self.z[indices]
        ####################################
        # adjust_points()
        norms = np.sqrt(self.x_selected ** 2 + self.y_selected ** 2 + self.z_selected ** 2)
        x_unit = self.x_selected / norms
        y_unit = self.y_selected / norms
        z_unit = self.z_selected / norms
        self.x_adjusted = self.x_selected + x_unit * vector
        self.y_adjusted = self.y_selected + y_unit * vector
        self.z_adjusted = self.z_selected + z_unit * vector
        ####################################
        # find_centroid()
        adjusted_points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
        self.centroid = np.mean(adjusted_points, axis=0)

    def get_centroid(self):
        return self.centroid


if __name__ == "__main__":
    print('Error: not main file')
