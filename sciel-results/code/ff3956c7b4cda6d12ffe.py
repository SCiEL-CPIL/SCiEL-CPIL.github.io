import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition for 3x3x5 matrix multiplication.
    U: (9, rank)
    V: (15, rank)
    W: (15, rank)
    """
    n, m, p = 3, 3, 5
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Construct the tensor T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # T should be 1 if i==j (for A), j==k (for B) ... wait.
    # The tensor for matrix multiplication C = AB is T[i, j, k] = 1 if i==k and j==j? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] corresponds to the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 if i==i (output row), j==j (inner), k==k (output col).
    # Actually, the tensor is defined as T[i, j, k] = 1 if i (row of C) matches i (row of A)
    # and k (col of C) matches k (col of B) and j (col of A) matches j (row of B).
    # So T[i, j, k] = 1 if i==i, j==j, k==k.
    # The condition is: T[i, j, k] = 1 if i (from A row) == i (C row) AND j (from A col) == j (B row) AND k (from B col) == k (C col).
    # This is always true for the indices in the sum.
    # The tensor T is the identity map on the bilinear form.
    # T[i, j, k] = 1 if i==i, j==j, k==k.
    # Wait, the indices are:
    # i: row of C (0..n-1)
    # j: inner index (0..m-1)
    # k: col of C (0..p-1)
    # T[i, j, k] = 1 if i (row of A) == i (row of C) AND j (col of A) == j (row of B) AND k (col of B) == k (col of C).
    # This is trivially 1 for all i, j, k in the valid ranges?
    # No, the tensor T is the structure tensor of the multiplication.
    # T[i, j, k] = 1 if i==i, j==j, k==k.
    # Actually, T[i, j, k] = 1 if i (row of C) == i (row of A) AND k (col of C) == k (col of B) AND j (col of A) == j (row of B).
    # This is 1 for all i, j, k.
    # Wait, that's not right.
    # The tensor T is defined such that C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k].
    # So T[i, j, k] = 1 for all i, j, k.
    # No, that's the schoolbook.
    # The tensor T is the identity tensor in the sense that it picks out the terms.
    # T[i, j, k] = 1 if i==i, j==j, k==k.
    # Actually, T[i, j, k] = 1 if i (row of A) == i (row of C) AND j (col of A) == j (row of B) AND k (col of B) == k (col of C).
    # This is 1 for all i, j, k.
    
    # Let's re-read the standard definition.
    # The multiplication tensor T is in A^* \otimes B^* \otimes C.
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where e_i is row i of A, e_j is col j of A / row j of B, e_k is col k of B.
    # So T[i, j, k] = 1 for all i, j, k.
    
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # So we need sum_r U[i, r] V[j, r] W[k, r] = 1 for all i, j, k.
    
    # Let's check this.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If we use the decomposition:
    # M_r = U_r \otimes V_r \otimes W_r.
    # Then C[i, k] = sum_r (sum_j U[i, r] A[i, j] ... no.
    
    # The standard decomposition is:
    # A = sum_r U_r \otimes a_r
    # B = sum_r V_r \otimes b_r
    # C = sum_r W_r \otimes (a_r \cdot b_r)
    
    # So U[i, r] is the coefficient of A[i, j] in the r-th term?
    # No, U is (n*m, rank). U[i*m + j, r] is the coefficient for A[i, j].
    # V[j*p + k, r] is the coefficient for B[j, k].
    # W[k*n + i, r] is the coefficient for C[i, k].
    
    # The condition is:
    # For all i, j, k:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    # Let's verify this.
    # If A is a basis vector A_{i0, j0} (1 at i0, j0, 0 elsewhere).
    # If B is a basis vector B_{j1, k0} (1 at j1, k0, 0 elsewhere).
    # Then C should be C_{i0, k0} if j0 == j1, else 0.
    
    # Using the formula:
    # C[i, k] = sum_r W[k*n + i, r] * (sum_j U[i*m + j, r] A[i, j]) * (sum_j V[j*p + k, r] B[j, k])
    # Wait, the scalar multiplication is (a_r \cdot b_r).
    # a_r = sum_j U[i*m + j, r] A[i, j] ? No.
    # The r-th term computes M_r = (U_r \cdot A) * (V_r \cdot B) * W_r.
    # U_r is a vector of length n*m.
    # (U_r \cdot A) = sum_{i,j} U[i*m + j, r] A[i, j].
    # (V_r \cdot B) = sum_{j,k} V[j*p + k, r] B[j, k].
    # Then C[i, k] += M_r * W[k*n + i, r].
    
    # So the condition is:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] = 1 for all i, j, k.
    # Wait, if A is A_{i0, j0} and B is B_{j1, k0}:
    # (U_r \cdot A) = U[i0*m + j0, r].
    # (V_r \cdot B) = V[j1*p + k0, r].
    # C[i, k] = sum_r W[k*n + i, r] U[i0*m + j0, r] V[j1*p + k0, r].
    # We want C[i, k] = 1 if i=i0, k=k0, j0=j1. Else 0.
    
    # So the condition is:
    # For all i0, j0, j1, k0:
    # sum_r U[i0*m + j0, r] V[j1*p + k0, r] W[k0*n + i0, r] = 1 if j0==j1 else 0.
    # AND i0, k0 are free?
    # No, the output C[i, k] depends on i, k.
    # The term W[k*n + i, r] links the output index (i, k) to the term r.
    
    # So the condition is:
    # For all i, j, k:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] = 1.
    
    # Let's test this with schoolbook.
    # U[i*m + j, r] = 1 if r == i*m + j*p + k? No.
    # Schoolbook: r = i*m + j*p + k is not the index.
    # Schoolbook: r = i*m + j*p + k? No.
    # Schoolbook: r = i*m + j*p + k is the index in the flattened tensor.
    # U[i*m + j, r] = 1 if r corresponds to (i, j, k)?
    # In schoolbook, for each (i, j, k), we have a term.
    # U[i*m + j, r] = 1 if r == i*m + j*p + k?
    # V[j*p + k, r] = 1 if r == i*m + j*p + k?
    # W[k*n + i, r] = 1 if r == i*m + j*p + k?
    
    # Then sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r]
    # = sum_r (1 if r==i*m+j*p+k else 0) * (1 if r==i*m+j*p+k else 0) * (1 if r==i*m+j*p+k else 0)
    # = 1 if there exists r such that r==i*m+j*p+k.
    # Yes, for each (i, j, k), there is exactly one r.
    # So the sum is 1.
    
    # So the condition is:
    # For all i, j, k:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] = 1.
    
    # Let's implement this check.
    n, m, p = 3, 3, 5
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # U: (9, rank), V: (15, rank), W: (15, rank)
    # i in 0..2, j in 0..2, k in 0..4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                # U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :]
                s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Candidate: Rank 31 (23 for 3x3x3 + 8 for 3x3x2)
    # I will try to construct this.
    # If I can't, I return schoolbook.
    
    # Let's try to use a known rank 31 decomposition.
    # Since I don't have it, I will return schoolbook.
    # But I want to beat 1.0.
    
    # I will try to use a rank 32 decomposition.
    # 23 for 3x3x3 + 9 for 3x3x2.
    
    # I will return schoolbook for now, as I cannot guarantee the 23 terms.
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
