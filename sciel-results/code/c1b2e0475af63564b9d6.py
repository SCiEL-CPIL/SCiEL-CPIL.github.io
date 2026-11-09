import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication
    of a 3x3 matrix by a 3x5 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # Using NumPy for efficient construction and to ensure structural distinctness
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Fill in the schoolbook decomposition terms
    # Each term c corresponds to indices (i, j, k)
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u, c] = 1
                V[row_v, c] = 1
                W[row_w, c] = 1
                c += 1
                
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    # Self-check
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Construction took {end - start:.4f}s")
    print(f"Rank: {len(res['U'][0])}")
    
    # Optional: Verify correctness for one random case
    # n, m, p = 3, 3, 5
    # A = np.random.randint(0, 10, (n, m))
    # B = np.random.randint(0, 10, (m, p))
    # C_expected = A @ B
    
    # C_computed = np.zeros((n, p))
    # U_arr = np.array(res['U'])
    # V_arr = np.array(res['V'])
    # W_arr = np.array(res['W'])
    
    # # The decomposition is: sum_c U[:,c] * V[:,c] * W[:,c]
    # # This forms a 3-tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] 
    # # Wait, the tensor for matrix multiplication C = A B is T[i, j, k] = A[i, l] B[l, k]
    # # The standard tensor decomposition for multiplication is:
    # # C[i, k] = sum_j sum_l A[i, j] B[j, l] delta_{k,l}? No.
    # # C[i, k] = sum_j A[i, j] B[j, k].
    # # The tensor T is indexed by (i, j, k) where i is row of C, j is col of A/row of B, k is col of C.
    # # T[i, j, k] = delta_{i, row_A} * delta_{j, col_A/row_B} * delta_{k, col_B}
    # # In the factorization: T = sum_c u_c * v_c * w_c
    # # u_c is in R^(n*m) -> indices (i, j_A)
    # # v_c is in R^(m*p) -> indices (j_B, k)
    # # w_c is in R^(p*n) -> indices (k, i) ?
    # # The task says: W[k*n+i, r]. This implies W is indexed by (k, i).
    # # So the term is U[i, j, c] * V[j, k, c] * W[k, i, c].
    # # Summing over j, k, c?
    # # No, the identity is:
    # # For each output C[i, k], we sum over j.
    # # The tensor T_{i, j, k} = 1 if it corresponds to the multiplication term.
    # # Actually, the matrix multiplication tensor T is defined such that:
    # # <T, A, B> = C
    # # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # # Wait, the standard form is T[i, j, k] with indices (row_out, row_in1, col_in2)?
    # # Let's check the convention in the prompt:
    # # "W[k*n+i, r]" -> Index r in W corresponds to (k, i).
    # # "V[j*p+k, r]" -> Index r in V corresponds to (j, k).
    # # "U[i*m+j, r]" -> Index r in U corresponds to (i, j).
    # # The contraction is usually sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    # # This sum over r gives a value for each (i, j, k).
    # # This value should be 1 if the "standard" term A[i,j]B[j,k] exists?
    # # No, the tensor T represents the operation C = AB.
    # # C[i, k] = sum_j A[i, j] B[j, k].
    # # So for a fixed i, k, the coefficient of A[i, j] B[j, k] is 1.
    # # The tensor T has indices (i, j, k). T[i, j, k] = 1.
    # # The decomposition is T = sum_r u_r * v_r * w_r.
    # # u_r[i, j], v_r[j, k], w_r[k, i].
    # # So sum_r u_r[i, j] v_r[j, k] w_r[k, i] should equal T[i, j, k].
    # # In schoolbook, for a specific (i, j, k), there is one r (c) such that
    # # u_c[i, j]=1, v_c[j, k]=1, w_c[k, i]=1.
    # # All other r' have at least one zero factor.
    # # So the sum is 1.
    # # This matches T[i, j, k] = 1 for all i, j, k?
    # # Yes, because for every combination of i, j, k, there is a term in the sum C[i,k] = sum_j A[i,j]B[j,k].
    # # Wait, C[i,k] does not depend on j directly in the output, but the term A[i,j]B[j,k] exists for each j.
    # # So T[i, j, k] = 1 for all i, j, k.
    # # The schoolbook decomposition provides exactly this.
    
    pass
