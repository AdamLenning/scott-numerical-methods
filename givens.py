import numpy as np
import random
import itertools
import copy
import operator
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#choleski > householder and dolittle (AB decopm) > gram shmidt on nxp


def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = np.linalg.norm([a, b])
    c = a/r
    s = -b/r

    return (c, s)

def givens_rotation_danbar(A):
    """Givens rotation found at: https://github.com/danbar/qr_decomposition/blob/master/qr_decomposition/qr_decomposition.py

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)
            

def main(V, step, reps):
    df = pd.DataFrame(columns=['i', 'time', 'exp'])

    for i in range(2, V, step):
            
        exp_sum = 0
        nonexp_sum = 0

        for rep in range(reps):

            mat = np.random.randint(10, size=(i, i)).astype(float)
            
            start_exp = time.time()
            givens_rotation_danbar(mat)
            end_exp = time.time()
            exp_sum += end_exp - start_exp

            # start_nonexp = time.time()
            # givens_rotation(mat, exp=False)
            # end_nonexp = time.time()
            # nonexp_sum += end_nonexp - start_nonexp

        exp_avg = exp_sum / reps
        nonexp_avg = nonexp_sum / reps


        df = df.append({'i': i, 'time': exp_avg, 'exp': True}, ignore_index=True)
        df = df.append({'i': i, 'time': nonexp_avg, 'exp': False}, ignore_index=True)

        print("finished", i)
        


    sns.lineplot(x=df.i, y=df.time, hue=df.exp)
    plt.xlabel("Size of matrix")
    plt.ylabel("Avg Time of {} trials (sec)".format(reps))
    plt.legend()
    plt.savefig('givens_time.png')


if __name__ == "__main__":
    # main(100, 5, 30)


    # mat = np.random.randint(10, size=(4, 4)).astype(float)
    mat = np.array([[3, 4, 2, 1],
                    [5, 6, 7, 8],
                    [3, 4, 5, 1],
                    [1, 2, 3, 4]], dtype=float)

    # expected answer:  6.6332495807108	8.442317648177381	8.894584665044027	7.53778361444409
    #                   0	            0.8528028654224415	1.0660035817780529	0.42640143271122266
    #                   0	            0	                2.5980762113533156	2.8867513459481278
    #                   0	            0	                0	                4.0824829046386295

    r = givens_rotation_danbar(mat)[1]
    q = givens_rotation_danbar(mat)[0]
    # r_inverse = np.diag(1 / np.sqrt(np.diag(r)))
    # print("r' r", (r_inverse @ r).T @ r)
    # print('r', r)
    # print('q', q)
    print(q @ r)
    # print("r^t r", r.T @ r)
    # print(givens_rotation(mat, exp=True))



