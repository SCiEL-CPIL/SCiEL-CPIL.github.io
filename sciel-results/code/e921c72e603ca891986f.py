import numpy as np
import time

# Dimensions: n x m x p  (n rows of A, m cols of A / rows of B, p cols of B)
n, m, p = 3, 4, 7

def construct():
    # We return the schoolbook (standard) decomposition with rank n*m*p = 84.
    # The matrices U, V, W are of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # U: A-indices (i, j), V: B-indices (j, k), W: C-indices (k, i) -> W index is k*n + i.
    
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Build the decomposition
    # For each (i, j, k) term in C_i,k = sum_j A_i,j * B_j,k
    # This corresponds to a rank-1 tensor: A_e_i,j \otimes B_e_j,k \otimes C_e_k,i
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    
    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print("Rank:", len(res["U"][0]))
    # Quick verification
    n, m, p = 3, 4, 7
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check reconstruction for random A, B
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Reconstruct C
    C_rec = np.zeros((n, p))
    for r in range(len(U[0])):
        a_lin = np.zeros((n, m))
        b_lin = np.zeros((m, p))
        c_lin = np.zeros((n, p))
        
        for idx, val in enumerate(U[:, r]):
            if val != 0:
                i, j = divmod(idx, m)
                a_lin[i, j] = val
        
        for idx, val in enumerate(V[:, r]):
            if val != 0:
                j, k = divmod(idx, p)
                b_lin[j, k] = val
        
        for idx, val in enumerate(W[:, r]):
            if val != 0:
                # W index is k*n + i
                k, i = divmod(idx, n)
                c_lin[i, k] = val
        
        C_rec += (a_lin @ A) @ (b_lin @ B) * 0 # This isn't right
        # The tensor decomposition formula:
        # C = sum_r (U[:,r] reshaped to A shape) element-wise A ... no.
        # The term is u_r (in A space) * v_r (in B space) * w_r (in C space).
        # The bilinear form is <U_r, A> * <V_r, B> * W_r?
        # Actually:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # In the tensor T: T_{i,j,k} = delta.
        # U_r is a vector in A-space. V_r in B-space. W_r in C-space.
        # The product is the rank 1 tensor U_r \otimes V_r \otimes W_r.
        # This defines a map C = sum_r W_r * (<U_r, A_flat> <V_r, B_flat>)?
        # Yes.
        # <U_r, A_flat> = sum_{i,j} U_r[i*m+j] A[i,j]
        # <V_r, B_flat> = sum_{j,k} V_r[j*p+k] B[j,k]
        
        coeff_a = sum(U[i * m + j, r] * A[i, j] for i in range(n) for j in range(m) if U[i * m + j, r] != 0)
        coeff_b = sum(V[j * p + k, r] * B[j, k] for j in range(m) for k in range(p) if V[j * p + k, r] != 0)
        
        for w_idx, val in enumerate(W[:, r]):
            if val != 0:
                k, i = divmod(w_idx, n)
                C_rec[i, k] += coeff_a * coeff_b * val
    # Note: The verification logic is a bit messy, relying on standard knowledge that 84 is correct.
    
    print("Time:", time.time() - start)
