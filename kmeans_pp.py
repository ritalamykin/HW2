import sys
import mykmeanssp as km
import numpy as np
import pandas as pd
global k, max_iter, eps, file_name_1, file_name_2, d, N


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


# if we need to check k is in range check here
def preprocessing_files():
    """
    reading file_name_1, file_name_2 (txt/csv files), combine both with inner join by index column
    :return: np array (combined)
    """
    global d, N
    arr1 = np.genfromtxt(file_name_1, delimiter=',', dtype=np.float64)
    arr2 = np.genfromtxt(file_name_2, delimiter=',', dtype=np.float64)
    df1 = pd.DataFrame(arr1[:, 1:], index=arr1[:, 0], dtype=np.float64)
    df2 = pd.DataFrame(arr2[:, 1:], index=arr2[:, 0], dtype=np.float64)
    data = df1.join(df2, how="inner", rsuffix='r')
    data.sort_index(inplace=True)
    d = len(data.columns)
    N = len(data)
    if k < 1 or k > N:
        print('Invalid Input!')
        exit(1)
    return data.to_numpy(dtype=np.float64)


def kmeans(points):
    """
    centroids:
    :param points:
    :return:
    """
    centroids = np.zeros(k, dtype=np.int64)
    d_val = np.zeros(N, dtype=np.float64)
    p_val = np.zeros(N, dtype=np.float64)
    np.random.seed(0)
    centroids[0] = np.random.choice(N)
    i = 1
    while i < k:
        for m in range(N):
            min_dist = min([points[m, :]-points[idx, :] for idx in centroids[:i]], key=lambda x: np.power(np.linalg.norm(x), 2))
            d_val[m] = np.power(np.linalg.norm(min_dist), 2)
        for m in range(N):
            p_val[m] = d_val[m]/np.sum(d_val)
        centroids[i] = np.random.choice(N, p=p_val)
        i += 1
    return centroids


def create_kmeans_arguments(centroids, points):
    res = [float(d), float(k), float(eps), float(N), float(max_iter)]
    res.extend(centroids.flatten().tolist())
    res.extend(points.flatten().tolist())
    return res


def create_final_string(centroids, indices):
    final_str = [np.array2string(indices, separator=',')[1:-1].replace("\n", "")]
    for i in range(k):
        c = centroids[i]
        c_str = np.array2string(c, formatter={'float_kind': lambda x: "%.4f" % x}, separator=',')[1:-1].replace("\n", "")
        final_str.append(c_str)
    return '\n'.join(final_str).replace(" ", "")


def main():
    cmd_input()
    points = preprocessing_files()
    indices = kmeans(points)
    centroids_from_c = km.fit(create_kmeans_arguments(points[indices], points))
    centroids = np.array(centroids_from_c).reshape(k, d)
    final_string = create_final_string(centroids, indices)

    print(final_string)


if __name__ == "__main__":
    main()
