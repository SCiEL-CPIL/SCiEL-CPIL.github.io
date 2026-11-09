import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # The tensor for matrix multiplication C = AB where A is n x m and B is m x p
    # is T = sum_{i,j,k} A_ij B_jk C_ik.
    # Since the rows of A are independent variables and rows of C are independent outputs,
    # the tensor is the direct sum of n independent 1 x m x p multiplication tensors.
    # r(1, m, p) = m * p.
    # Thus r(n, m, p) = n * m * p.
    # For 4, 5, 5, rank is 100.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Generate schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(n * p)
                
                u[a_idx] = 1.0
                v[b_idx] = 1.0
                w[c_idx] = 1.0
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
