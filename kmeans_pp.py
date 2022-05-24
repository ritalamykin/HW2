import sys

import numpy as np
import pandas as pd

global k, max_iter, eps, file_name_1, file_name_2


def cmd_input():
    global k, max_iter, eps, file_name_1, file_name_2
    args_len = len(sys.argv)
    k = sys.argv[1]
    if args_len == 6:
        max_iter = sys.argv[2]
        eps = sys.argv[3]
        file_name_1 = sys.argv[4]
        file_name_2 = sys.argv[5]
    else:
        max_iter = '300'
        eps = sys.argv[2]
        file_name_1 = sys.argv[3]
        file_name_2 = sys.argv[4]
    valid_input()


def valid_input():
    """
    checking input type is valid (numeric/ float)
    :return: exit(1) with error
    """
    global k, max_iter, eps
    try:
        k = int(k)
        max_iter = int(max_iter)
        eps = float(eps)
    except ValueError:
        print("Invalid Input!")
        exit(1)


class Point:
    def __init__(self, coords, index, d=0, p=0):
        self.coords = coords
        self.index = index
        self.d = d
        self.p = p

    def distance(self, point2):
        return sum([(p - q) ** 2 for p, q in zip(self.coords, point2)])


# if we need to check k is in range check here
def preprocessing_files():
    """
    reading file_name_1, file_name_2 (txt/csv files), combine both with inner join by index column
    :return: np array (combined)
    """
    arr1 = np.genfromtxt(file_name_1, delimiter=',', dtype=np.float64)
    arr2 = np.genfromtxt(file_name_2, delimiter=',', dtype=np.float64)
    df1 = pd.DataFrame(arr1[:, 1:], index=arr1[:, 0], dtype=np.float64)
    df2 = pd.DataFrame(arr2[:, 1:], index=arr2[:, 0], dtype=np.float64)
    data = df1.join(df2, how="inner", rsuffix='r')
    data = data.apply(lambda x: Point(list(x), x.name), axis=1).reset_index(drop=True)
    return data.to_numpy(dtype=Point)


def kmeans(points):
    """
    centroids:
    :param points:
    :return:
    """
    np.random.seed(0)
    centroids = np.empty(k, dtype=Point)
    centroids[0] = np.random.choice(points, 1, replace=False)[0]
    # centroids[0] = data_arr[np.random.choice(len(data_arr), 1, replace=False)]
    i = 1
    while i < k:
        sum_d = 0
        for point in points:
            point.d = min([point.distance(centroids[j].coords) for j in range(i)])
            sum_d += point.d
        for point in points:
            point.p = point.d/sum_d
        centroids[i] = np.random.choice(points, 1, p=np.vectorize(lambda x: x.p)(points))[0]
        i += 1
    return centroids


def main():
    cmd_input()
    # global k, max_iter, eps, file_name_1, file_name_2
    # k =3
    # max_iter = 300
    # eps = 0.01
    # file_name_1 = 'test_data/input_1_db_1 copy.csv'
    # file_name_2 = 'test_data/input_1_db_2.txt'
    points = preprocessing_files()
    centroids = kmeans(points)
    print([point.coords for point in centroids])


if __name__ == "__main__":
    main()









