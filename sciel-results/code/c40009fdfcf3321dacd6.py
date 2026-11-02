import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2
    add_strassen(0, 1, 0)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # Check for linear dependencies to reduce rank
    if time.time() < deadline - 10.0:
        terms = []
        for r in range(rank):
            t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
            terms.append(t.flatten())
            
        M = np.array(terms).T
        
        # Use a tighter tolerance to catch near-dependencies
        rank_M = np.linalg.matrix_rank(M, tol=1e-8)
        
        if rank_M < rank:
            # Find null vector
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            c = Vt[-1, :]
            k_idx = np.argmax(np.abs(c))
            
            if np.abs(c[k_idx]) > 1e-8:
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(U_mat[:, r] * coeff)
                    new_V.append(V_mat[:, r] * coeff)
                    new_W.append(W_mat[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                # Verify the reduced decomposition
                if time.time() < deadline - 5.0:
                    terms_new = []
                    for r in range(len(new_U.T)):
                        t = np.einsum('i,j,k->ijk', new_U[:, r], new_V[:, r], new_W[:, r])
                        terms_new.append(t.flatten())
                    M_new = np.array(terms_new).T
                    M_orig = np.array(terms).T
                    # Check if M_new is close to M_orig (up to column permutation)
                    # Actually, we just need to check if the tensor is preserved
                    # Sum of new terms should equal sum of old terms
                    sum_new = np.sum(M_new, axis=1)
                    sum_old = np.sum(M_orig, axis=1)
                    if np.allclose(sum_new, sum_old, atol=1e-8):
                        return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
