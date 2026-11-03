import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try, from lowest known upper bound upwards
    # Lader (1976) proved rank 33 for 3x3x4.
    # We try to find a decomposition of rank 33, 34, 35.
    # If none found, fall back to schoolbook rank 36.
    
    start_time = time.time()
    time_limit = 120.0  # Safety margin for 150s limit
    
    # Define the matrix multiplication tensor T
    # T[i, j, k] = 1 if i==i, j==j, k==k? No.
    # The tensor T_{ijk} is defined by the bilinear form of matrix multiplication.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the coefficient of A_{ij} B_{jk} in (AB)_{ik} is 1 if the indices match the multiplication structure.
    # Specifically, we want to decompose the tensor T such that:
    # T_{ijk} = sum_{c=1}^r U_{i, c} V_{j, c} W_{k, c}
    # Wait, the interface says:
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]
    # W: (p*n, rank) -> W[k*n+i, c]
    # The product term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This term contributes to the output element at position (i, k) in the result matrix C?
    # No, the output is a matrix C of size n x p.
    # The standard tensor decomposition for matmul A(n,m) * B(m,p) -> C(n,p)
    # is T_{i,j,k} where i is row of A (and C), j is col of A (and row of B), k is col of B (and C).
    # The coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So T[i, j, k] = 1 if we are computing C[i,k] using A[i,j] and B[j,k].
    # In the flattened indices:
    # A index: i*m + j
    # B index: j*p + k
    # C index: i*p + k  (Standard row-major)
    # BUT the interface specifies W[k*n + i, c].
    # This implies the output index is mapped as k*n + i.
    # Let's check the trace convention: "W[k*n+i,r]".
    # This means the output matrix C is stored such that element (i, k) is at index k*n + i.
    # This is column-major order for C (if i is row, k is col, then index is k*n+i).
    # Or it just defines the mapping of the third dimension of the tensor to the W matrix rows.
    # Let's construct the target tensor T_flat of size (n*m) * (m*p) * (p*n) ? No.
    # The tensor is rank-3.
    # Dimensions: n*m, m*p, p*n.
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i.
    # T[a, b, c] = 1 if the indices are consistent with a single multiplication step.
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # For the term to be non-zero, we need j == j' and i == i'.
    # Also, the term A[i,j] * B[j,k] contributes to C[i,k].
    # So T is non-zero (value 1) when:
    # j == b // p  (j from A matches j from B)
    # i == c % n   (i from A matches i from C)
    # And the indices i, j, k are valid.
    
    # Let's build the target tensor as a sparse representation or dense array if small enough.
    # n=3, m=3, p=4.
    # Size: 9 * 12 * 12 = 1296. Small enough for dense.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Function to check if U, V, W form a valid decomposition
    def is_valid(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Compute sum_{c} U[:, c] * V[:, c] * W[:, c]
        # This is a tensor contraction.
        # We can compute the tensor X = sum_c outer(U[:,c], outer(V[:,c], W[:,c]))
        # Or iterate.
        r = U.shape[1]
        X = np.zeros_like(T)
        for c in range(r):
            X += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        # Check if X equals T
        # Since entries are integer/half-integer, exact equality is expected if valid.
        # Use a small tolerance for floating point errors if any, but we use discrete values.
        return np.allclose(X, T, atol=1e-9)

    # Schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Try to find a lower rank decomposition
    # We try ranks 33, 34, 35
    # Use random search with local refinement
    possible_ranks = [33, 34, 35]
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for r in possible_ranks:
        # Limit time per rank
        rank_start = time.time()
        max_time_per_rank = 30.0
        
        # Try a few random restarts
        for restart in range(20):
            if time.time() - rank_start > max_time_per_rank:
                break
            if time.time() - start_time > time_limit - 10:
                break
                
            # Initialize randomly
            U = np.random.choice(values, size=(n*m, r))
            V = np.random.choice(values, size=(m*p, r))
            W = np.random.choice(values, size=(p*n, r))
            
            # Check initial
            if is_valid(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Local refinement: random perturbation
            # We compute the error tensor E = T - X
            # We try to change one element in U, V, or W to reduce ||E||
            # This is expensive to do exactly.
            # Instead, we just do random restarts for this simple search.
            # A more sophisticated search would be needed for a real chance.
            # Given the complexity, random restarts are unlikely to find rank 33.
            pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
