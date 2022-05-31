from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def plot(data_dict):
    plt.figure()
    plt.plot(list(data_dict.keys()), list(data_dict.values()))
    plt.xticks(list(data_dict.keys()))
    plt.axvspan(2.5, 3.5, color='red', alpha=0.2)
    plt.ylabel("Inertia")
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.savefig('elbow.png')


def main():
    data = load_iris().data
    points_arr = np.array(data)
    inertia_d = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(points_arr)
        inertia_d[k] = kmeans.inertia_
    plot(inertia_d)


if __name__ == '__main__':
    main()
