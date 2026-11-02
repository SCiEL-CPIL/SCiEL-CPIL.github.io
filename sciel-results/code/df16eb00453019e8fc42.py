# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # Generate the naive 120 rank-1 terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u is vector in R^{n*m}
                # v is vector in R^{m*p}
                # w is vector in R^{p*n}
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                # Flatten the outer product u \otimes v \otimes w
                # Shape (n*m, m*p, p*n)
                tensor = np.outer(np.outer(u, v), w)
                vec = tensor.flatten()
                terms.append(vec)
    
    # We have 120 vectors in R^{14400}
    # Check if any vector is in the span of the others
    # This is a rank check.
    # If the rank of the 120 vectors is < 120, we can reduce the number of terms.
    
    # Construct the matrix of vectors as rows
    M = np.array(terms)
    
    # Compute the rank
    # SVD is robust for rank determination
    try:
        rank = np.linalg.matrix_rank(M)
    except Exception:
        rank = len(terms)
        
    # If rank < 120, we need to find a basis.
    # For now, if rank is 119, we can drop one linearly dependent vector.
    # However, the decomposition must be EXACT.
    # If we drop a vector, the sum of the remaining vectors must still equal the target tensor.
    # The target tensor T is the sum of all 120 vectors.
    # If v_120 = sum c_i v_i, then T = sum_{i=1}^{119} v_i + v_120 = sum_{i=1}^{119} (1+c_i) v_i.
    # So we can just adjust the coefficients of the remaining vectors.
    # But our interface requires integer/half-integer entries in U, V, W.
    # The coefficients c_i might be rational.
    
    # If the rank is 120, we cannot reduce it this way.
    # Let's assume the rank is 120 for the naive set.
    
    # Return the naive decomposition as a fallback, but formatted as U, V, W
    # We will return the schoolbook decomposition which is rank 120.
    # To potentially improve, we would need to find a better basis, which is hard.
    
    rank_val = n * m * p
    U = [[0] * rank_val for _ in range(n * m)]
    V = [[0] * rank_val for _ in range(m * p)]
    W = [[0] * rank_val for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
