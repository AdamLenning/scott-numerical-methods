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




def my_plot(V, step, reps):

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
            

def main():
    mat_part = np.array([[3, 4, 2, 1],
                    [5, 6, 7, 8],
                    [3, 4, 5, 1]], dtype=float)
    mat_add = np.array([1, 2, 3, 4], dtype=float)
    
    mat_full = np.array([[3, 4, 2, 1],
                    [5, 6, 7, 8],
                    [3, 4, 5, 1],
                    [1, 2, 3, 4]], dtype=float)
    print("mat\n", mat_full)
    
    r = givens_rotation_danbar(mat_full)[1]
    # q = givens_rotation_danbar(mat)[0]

    # r_inverse = np.diag(1 / np.sqrt(np.diag(r)))
    # print("r' r", (r_inverse @ r).T @ r)
    # print('r', r)
    # print('q', q)
    
    print("full\n", r)

    r_part = givens_rotation_danbar(mat_part)[1]
    r_add = givens_rotation_danbar(mat_add)[1]
    r_part = np.vstack([r_part, r_add])
    r_added = givens_rotation_danbar(r_part)[1]
    print("added\n", r_added)

    # permutation = [1, 2, 3, 0]
    # idx = np.empty_like(permutation)
    # idx[permutation] = np.arange(len(permutation))
    # r_ordered = r[:, idx]
    # print("r_ordered\n", r_ordered)
    # r_reord = givens_rotation_danbar(r_ordered)[1]
    # print("r_r_ordered\n", r_reord)



    # permutation = [1, 2, 3, 0]
    # idx = np.empty_like(permutation)
    # idx[permutation] = np.arange(len(permutation))
    # ordered = mat_full[:, idx]
    # print("raw_ordered\n", ordered)
    # r_reord1 = givens_rotation_danbar(ordered)[1]
    # print("r_raw_reordered\n", r_reord1)

    # print(np.array_equal(r_reord, r_reord1))

    # ## undo permute
    # undo_perm = [3, 1, 0, 2]
    # idx = np.empty_like(undo_perm)
    # idx[undo_perm] = np.arange(len(undo_perm))
    # reord = r_reord[:, idx]
    # print("reord\n", reord)
    # r_undid = givens_rotation_danbar(reord)[1]
    # print("r_undid\n", r_undid)


if __name__ == "__main__":
    main()

    # expected answer:  6.6332495807108	8.442317648177381	8.894584665044027	7.53778361444409
    #                   0	            0.8528028654224415	1.0660035817780529	0.42640143271122266
    #                   0	            0	                2.5980762113533156	2.8867513459481278
    #                   0	            0	                0	                4.0824829046386295

    



