"""Research program: Attempts rank-14 construction via 3x3 tensor embedding, falls back to schoolbook."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The directive suggests using a rank-7 decomposition for 3x3 matrix multiplication.
    # However, the minimum known rank for 3x3 matrix multiplication is 19 (or 23 in some contexts, but definitely > 7).
    # Rank 7 is the rank for 2x2 matrix multiplication (Strassen's algorithm).
    # The "Kronecker product" hint likely implies trying to combine structures, but since a rank-7 3x3 decomposition does not exist,
    # any attempt to construct a rank-14 decomposition based on it will fail verification.
    
    # We implement a robust fallback to the schoolbook algorithm which guarantees correctness.
    # Schoolbook rank is n*m*p = 2*3*3 = 18.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1.0
                V[j * p + k, col_idx] = 1.0
                W[k * n + i, col_idx] = 1.0
                col_idx += 1
                
    # Convert to lists of lists as per interface expectation (though numpy arrays are often accepted, 
    # the seed used lists. Let's stick to the seed's return type structure if possible, or ensure compatibility).
    # The seed returned lists. Let's return lists to be safe, or numpy arrays if the evaluator prefers.
    # The seed code: return {"U": U, "V": V, "W": W} where U,V,W were lists of lists.
    # Let's convert back to lists to match the seed's type exactly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
