import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # The tensor for multiplying a (n x m) matrix A by a (m x p) matrix B
    # has components T[i, j, k, l] = delta_{j,k}.
    # This tensor is separable along the column index l of B (and C).
    # Specifically, T[i, j, k, l] = delta_{j,k} * 1_l.
    # We can decompose this using rank m = 2 by fixing j (the shared index).
    # For each r in {0, 1} (corresponding to j = r):
    #   U[i, j, r] = delta_{i, 0} * delta_{j, r}  (or any non-zero factor, but let's use standard)
    #   Actually, let's look at the factorization:
    #   T[i, j, k, l] = sum_r U[i, j, r] V[k, l, r] W[l, i, r]
    #   Wait, the standard decomposition of matrix multiplication M_{n,m,p} is:
    #   T[i, j, k, l] = delta_{j,k}
    #   We can write delta_{j,k} = sum_{r=0}^{m-1} delta_{j,r} delta_{k,r}
    #   So let U[i, j, r] = delta_{j,r} (independent of i? No, U must depend on i)
    #   Let's check the dimensions.
    #   U is (n*m, rank). Index (i*m+j, r).
    #   V is (m*p, rank). Index (k*p+l, r).
    #   W is (p*n, rank). Index (l*n+i, r).
    #   
    #   We want: sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] = delta_{j,k}
    #   
    #   Let rank = m = 2. Let r index the value of j (and k).
    #   For a fixed r, let j=r and k=r.
    #   We need the sum over r to give 1 if j=k, 0 otherwise.
    #   If we set:
    #   U[i*m+j, r] = delta_{j,r}  (so U is 1 if j=r, else 0. Independent of i)
    #   V[k*p+l, r] = delta_{k,r}  (so V is 1 if k=r, else 0. Independent of l)
    #   W[l*n+i, r] = 1            (so W is always 1)
    #   Then sum_r delta_{j,r} * delta_{k,r} * 1 = delta_{j,k}.
    #   This works!
    #   
    #   Let's verify the shapes and indices.
    #   n=2, m=2, p=12.
    #   rank = 2.
    #   U shape: (4, 2).
    #     i=0, j=0 -> idx 0. U[0, r] = delta_{0,r}. U[0,0]=1, U[0,1]=0.
    #     i=0, j=1 -> idx 1. U[1, r] = delta_{1,r}. U[1,0]=0, U[1,1]=1.
    #     i=1, j=0 -> idx 2. U[2, r] = delta_{0,r}. U[2,0]=1, U[2,1]=0.
    #     i=1, j=1 -> idx 3. U[3, r] = delta_{1,r}. U[3,0]=0, U[3,1]=1.
    #   
    #   V shape: (24, 2).
    #     k=0, l=0..11 -> idx 0..11. V[idx, r] = delta_{0,r}. V[idx,0]=1, V[idx,1]=0.
    #     k=1, l=0..11 -> idx 12..23. V[idx, r] = delta_{1,r}. V[idx,0]=0, V[idx,1]=1.
    #   
    #   W shape: (24, 2).
    #     l=0..11, i=0..1.
    #     W[l*n+i, r] = 1.
    #     So all entries of W are 1.
    #   
    #   Let's double check the "trace convention" for W mentioned in the prompt:
    #   "The output factor follows the official trace convention W[k*n+i,r]"
    #   Wait, the prompt says W[k*n+i,r]. Usually C is p x n.
    #   Indices for C are (l, i) where l is row (0..p-1) and i is col (0..n-1).
    #   The prompt says W[k*n+i,r]. Here k likely corresponds to l (the row of C, which is the col of B).
    #   In the previous code, W indices were `w_00 = k_start * n + 0`.
    #   Here k_start is the column index of B, which is the row index of C.
    #   So the index is `l * n + i`.
    #   My derivation used W[l*n+i, r]. This matches.
    #   
    #   So the rank is 2.
    #   This is significantly better than 42.
    
    rank = m  # which is 2
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.ones((p * n, rank), dtype=int)
    
    # Fill U: U[i*m + j, r] = delta_{j, r}
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1  # r=j
    
    # Fill V: V[k*p + l, r] = delta_{k, r}
    for k in range(m):
        for l in range(p):
            V[k * p + l, k] = 1  # r=k
    
    # W is already all ones.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
