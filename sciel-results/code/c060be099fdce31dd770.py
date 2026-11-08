import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Build the matrix multiplication tensor T[i,j,k]
    # i = row_A * m + col_A (0..5)
    # j = row_B * p + col_B (0..11)
    # k = col_C * n + row_C (0..7)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Build the rank-22 block decomposition
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    def tensor_from_factors(U, V, W):
        r = U.shape[1]
        T_out = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T_out += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return T_out

    # Verify initial decomposition
    T_current = tensor_from_factors(U, V, W)
    if np.linalg.norm(T - T_current) > 1e-10:
        print("WARNING: Initial decomposition invalid")

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try removing each column to see if rank can be reduced
    for col_to_remove in range(rank):
        if time.time() - t_start > 140:
            break
        U_test = np.delete(U, col_to_remove, axis=1)
        V_test = np.delete(V, col_to_remove, axis=1)
        W_test = np.delete(W, col_to_remove, axis=1)
        T_test = tensor_from_factors(U_test, V_test, W_test)
        if np.linalg.norm(T - T_test) < 1e-10:
            best_rank = rank - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
