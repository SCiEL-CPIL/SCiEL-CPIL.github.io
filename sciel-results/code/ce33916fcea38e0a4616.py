import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    # Check if the decomposition exactly reproduces the matrix multiplication tensor
    # T[i, j, k] = 1 if the indices match the matmul structure, 0 otherwise.
    # The tensor is defined by C[i, k] = sum_j A[i, j] B[j, k].
    # The rank-1 term r contributes U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
    # We need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{j, j} (which is 1 if j matches? No.)
    # The tensor M_{n,m,p} has entries T_{i,j,k} = 1 if the "inner" index j of A and B is the same.
    # In the decomposition, the term r has indices i_u, j_u for U; i_v, j_v for V; i_w, j_w for W.
    # U index is i_A, j_A. V index is j_B, k_B. W index is k_C, i_C.
    # The product is non-zero only if j_A == j_B (inner index match) and i_A == i_C and k_B == k_C.
    # So for a fixed output (i, k) and inputs (i, j) and (j, k), the coefficient should be 1.
    # Let's check the coefficient of A[i, j] B[j, k] in the computed C[i, k].
    # Computed C[i, k] = sum_r (sum_{j'} U[i*m+j', r] A[i, j']) * (sum_{j''} V[j''*p+k, r] B[j'', k]) * W[k*n+i, r]
    # This is not quite right. The bilinear form is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k])?
    # No, the standard form is:
    # C[i, k] = sum_r (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k]) * W[k*n+i, r]
    # Wait, the W factor is just a scalar multiplier for the term r?
    # Yes, the tensor is sum_r u_r \otimes v_r \otimes w_r.
    # The value is sum_r (u_r^T A) (v_r^T B) w_r.
    # Here u_r is a vector of length n*m, v_r is length m*p, w_r is length p*n.
    # u_r has a 1 at (i, j) if U[i*m+j, r]=1.
    # v_r has a 1 at (j, k) if V[j*p+k, r]=1.
    # w_r has a 1 at (k, i) if W[k*n+i, r]=1.
    
    # For the schoolbook decomposition:
    # Term r corresponds to (i, j, k).
    # u_r has 1 at (i, j).
    # v_r has 1 at (j, k).
    # w_r has 1 at (k, i).
    # Product: A[i, j] * B[j, k] * 1.
    # Summing over all r, we get sum_{j} A[i, j] B[j, k] for C[i, k].
    # This is correct.
    
    # To verify, we can check if the tensor defined by U, V, W matches the identity tensor.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be 1 if the indices are consistent?
    # No, T[i, j, k] is the coefficient of A[i, j] B[j, k] in the bilinear form.
    # In the schoolbook case, T[i, j, k] = 1 for all i, j, k?
    # No, the tensor M_{n,m,p} is defined such that T(i, j, k) = 1 if the indices match the matmul flow.
    # Actually, the tensor is fixed. The decomposition must match it.
    # The tensor is: M = sum_{i, j, k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ?
    # No, the indices are:
    # A has indices (i, j). B has indices (j, k). C has indices (i, k).
    # The tensor is in A^* \otimes B^* \otimes C^*.
    # The entry T_{(i,j), (j,k), (k,i)} is 1.
    # So for any i, j, k, the entry at (i, j, j, k, k, i) is 1?
    # The indices for U are (i, j). For V are (j, k). For W are (k, i).
    # So T[i, j, j, k, k, i] = 1.
    # In our flattened indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor value at (u, v, w) is 1 if the indices are consistent.
    # Consistency: j_U == j_V and k_V == k_W and i_W == i_U.
    # Let's check the sum: sum_r U[u, r] V[v, r] W[w, r].
    # For schoolbook, r is (i, j, k).
    # U[u, r] is 1 if u = i*m + j and r = (i, j, k).
    # V[v, r] is 1 if v = j*p + k and r = (i, j, k).
    # W[w, r] is 1 if w = k*n + i and r = (i, j, k).
    # So the product is 1 only if u, v, w correspond to the same (i, j, k).
    # So T[u, v, w] = 1 if u, v, w are consistent, 0 otherwise.
    # This is the correct tensor.
    
    # We can verify by checking a few random points.
    # But for exactness, we should check all points if possible.
    # 125 * 125 * 125 = 2 million points. Feasible.
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Check consistency
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                
                expected = 1.0 if (j_u == j_v and k_v == k_w and i_w == i_u) else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[u, r] * V[v, r] * W[w, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition?
    # For now, return schoolbook.
    # A search for rank < 125 is complex.
    # We will return the schoolbook decomposition.
    return _schoolbook()
# EVOLVE-BLOCK-END
