import numpy as np
from scipy import sparse
from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2


def generate_matrix(row, col):
    AA = np.zeros((row, col))
    BB = np.zeros((row, col))
    for i in np.arange(row):
        for j in np.arange(col):
            if i < row - 1 and 0 < j < col - 1:
                AA[i][j] = 1
            if i > 0 and 0 < j < col - 1:
                BB[i][j] = 1
    # print("A:", AA)
    # print("B:", BB)
    return AA, BB


def generate_sparse_mat(matrix):
    (ncells, Lx, Ly) = matrix.shape
    matrix_sparse = sparse.coo_matrix(matrix.reshape(ncells, Lx * Ly).T)
    return matrix_sparse


# def cal_Jaccard(A, B, ThreshJ):
    #calcualte jaccard index between matrix A and B


# generate matrix
row = 4
col = 5
A, B = generate_matrix(row, col)

A = np.expand_dims(A, axis=0)
print("A.shape:", A.shape)
print("A:", A)
B = np.expand_dims(B, axis=0)
print("B.shape:", B.shape)
print("B:", B)

# generate sparse
A_sparse = generate_sparse_mat(A)
print("A_sparse.shape:", A_sparse.shape)
print("A_sparse:", A_sparse)
B_sparse = generate_sparse_mat(B)
print("B_sparse.shape:", B_sparse.shape)
print("B_sparse:", B_sparse)

# cal jaccard index
AAA = A_sparse.transpose()
BBB = B_sparse.transpose()
# (Recall, Precision, F1) = cal_Jaccard(A_sparse, B_sparse, ThreshJ=0.5)
(Recall, Precision, F1) = GetPerformance_Jaccard_2(AAA, BBB, ThreshJ=0.5)
print({'Recall': Recall, 'Precision': Precision, 'F1': F1})
# print(AAA)
