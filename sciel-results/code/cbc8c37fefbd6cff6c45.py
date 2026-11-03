import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 2, 5, 5  # 2x5 matrix times 5x5 matrix
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to find rank-minimal U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use the Kronecker product structure.
    # View A as a 2x1 block column of 1x5 rows.
    # The tensor for 2x5x5 multiplication can be decomposed using the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Direct construction:
    # For each (i, j, k), we need T[i,j,k] = 1.
    # We can use rank = n * m * p = 50 (schoolbook), but we can do better.
    
    # Key insight: The 2x5x5 multiplication tensor has a known lower bound.
    # For m x n x p matrix multiplication, the rank is at least max(mn, np, pm) = 25 here.
    # But we can use a smarter decomposition.
    
    # Let's try a structured decomposition:
    # U[i*m+j, r] where r encodes (i, j, k) in some compressed way
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # Approach 1: Try to find a low-rank decomposition using the algebraic structure
    # The tensor T[i,j,k] = 1 is a rank-1 tensor in the (i,j,k) space only if we sum over all indices.
    # Actually, T is the all-ones tensor, which has rank 1 as a 3D tensor!
    # Wait, no. The matrix multiplication tensor is NOT the all-ones tensor.
    # T[i,j,k] = 1 means: when multiplying A[i,j] * B[j,k], the result contributes to C[i,k].
    # The tensor entries are T[i,j,k] = 1 for all valid i,j,k.
    
    # So T is indeed the all-ones tensor of shape (n, m, p) = (2, 5, 5).
    # The rank of the all-ones tensor of shape (a, b, c) is 1!
    # Because T[i,j,k] = 1 = 1 * 1 * 1 for all i,j,k.
    
    # So we can set:
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k
    # W[k*n+i, 0] = 1 for all k, i
    # Then sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 * 1 * 1 = 1
    
    # This gives rank 1!
    
    # Let's verify:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With rank 1: S = U @ A_flat * V @ B_flat
    # U @ A_flat = sum_{i,j} U[i*m+j, 0] * A[i,j] = sum_{i,j} A[i,j] (since U is all 1s)
    # V @ B_flat = sum_{j,k} V[j*p+k, 0] * B[j,k] = sum_{j,k} B[j,k] (since V is all 1s)
    # S = (sum A) * (sum B)
    # C_rec[i,k] = W[k*n+i, 0] * S = 1 * (sum A) * (sum B)
    # But C[i,k] = sum_j A[i,j] * B[j,k] which is NOT generally equal to (sum A) * (sum B)
    
    # So the rank-1 decomposition does NOT work! The all-ones tensor interpretation is wrong.
    # The tensor T[i,j,k] = 1 means that the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    # The decomposition must satisfy:
    # sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_k V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # This is NOT simply T[i,j,k] = 1. The tensor structure is more complex.
    
    # Let me re-derive. The matrix multiplication tensor T is defined such that:
    # (A * B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # The rank-r decomposition requires:
    # sum_j A[i,j] * B[j,k] = sum_r U[i*m+j,r] ... no, that's not right either.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r sum_j U[i*m+j, r] A[i,j] * V[j*p+k, r] B[j,k] * W[k*n+i, r]
    
    # Wait, that's not the right form either. Let me think again.
    
    # The bilinear form: C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j*p+k', r] B[j,k']) * W[k*n+i, r]
    # This doesn't look right because the sums over j and k' are coupled.
    
    # The correct form is:
    # C[i,k] = sum_r S_r * W[k*n+i, r]
    # where S_r = (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])
    # But this still has j and k inside the same product, which is problematic.
    
    # Actually, the standard formulation is:
    # C[i,k] = sum_r (sum_j U_{ij,r} A_{ij}) * (sum_k V_{jk,r} B_{jk}) * W_{ki,r}
    # This requires that for each r, the sum over j in U and the sum over k in V are independent.
    # But U depends on j and V depends on j, so they are NOT independent.
    
    # I think I'm confusing myself. Let me use the seed program's formulation:
    # L = U @ A_flat where A_flat is row-major: A_flat[i*m+j] = A[i,j]
    # M = V @ B_flat where B_flat is row-major: B_flat[j*p+k] = B[j,k]
    # S = L * M (element-wise product)
    # C_rec[i,k] = sum_r W[k*n+i, r] * S[r]
    
    # For this to equal C[i,k] = sum_j A[i,j] B[j,k] for all A, B:
    # sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] A[i',j]) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    # = sum_j A[i,j] B[j,k]
    
    # This is a very complex condition. The seed program uses ALS to find U, V, W.
    
    # Let me just use the schoolbook construction which is known to work:
    # Rank = n * m * p = 2 * 5 * 5 = 50
    
    # For each (i, j, k):
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    # This gives:
    # L[col] = U[i*m+j, col] * A[i,j] = A[i,j] (for the specific i,j matching col)
    # M[col] = V[j*p+k, col] * B[j,k] = B[j,k] (for the specific j,k matching col)
    # S[col] = A[i,j] * B[j,k]
    # C_rec[i,k] = sum_{col} W[k*n+i, col] * S[col] = sum_{i',j,k'} W[k*n+i, i'*m+j, ...] ... 
    
    # In schoolbook, each col corresponds to a unique (i,j,k) triple.
    # W[k*n+i, col] = 1 only if the col's i matches and k matches.
    # So C_rec[i,k] = sum_{j} A[i,j] B[j,k] = C[i,k]. Correct!
    
    # Now, can we do better than 50?
    # The theoretical lower bound for 2x5x5 matrix multiplication rank is not well-known.
    # For 2x2x2, the rank is 7 (Strassen).
    # For 2x3x3, the rank is at least 16 (I think).
    # For 2x5x5, a reasonable guess is that the rank is significantly less than 50.
    
    # Let's try to find a better decomposition using the structure.
    # One approach: use the fact that n=2 is small.
    
    # Alternative: Use a block decomposition.
    # Write A as [A1; A2] where A1, A2 are 1x5 rows.
    # Write B as a 5x5 matrix.
    # C1 = A1 * B (1x5)
    # C2 = A2 * B (1x5)
    
    # The tensor for C1 = A1 * B is 1x5x5, which has rank 5 (schoolbook) or possibly less.
    # Similarly for C2.
    
    # If we can decompose 1x5x5 with rank R1, then 2x5x5 can be done with rank 2*R1.
    # The 1x5x5 multiplication: C[k] = sum_j A[j] B[j,k]
    # This is a vector-matrix product. The tensor is T[j,k] = 1 for all j,k.
    # Rank of 1x5x5 tensor: We need sum_r U[j,r] V[j,k,r] W[k,r] = 1 for all j,k.
    # This is the same as before but with n=1.
    # Schoolbook rank = 5.
    # Can we do better? For 1xn x nxn, the rank is at least n (I think), and schoolbook achieves n.
    # So 1x5x5 has rank 5.
    # Then 2x5x5 has rank at most 2*5 = 10.
    
    # Let's construct a rank-10 decomposition:
    # For i=0: use factors U0, V0, W0 for the 1x5x5 problem
    # For i=1: use factors U1, V1, W1 for the 1x5x5 problem
    # Combine: U = [U0; U1] (10 x 10), V = [V0; V1] (25 x 10), W = [W0; W1] (10 x 10)
    
    # Wait, the shapes need to match.
    # U has shape (n*m, R) = (10, R)
    # V has shape (m*p, R) = (25, R)
    # W has shape (p*n, R) = (10, R)
    
    # For i=0: U0 has shape (5, R0) where R0 is the rank for 1x5x5
    # For i=1: U1 has shape (5, R1)
    # If R0 = R1 = 5, then R = 10.
    
    # U[i*m+j, r] where i in {0,1}, j in {0,...,4}, r in {0,...,9}
    # For r = 0..4 (i=0 block): U[0*m+j, r] = U0[j, r], U[1*m+j, r] = 0
    # For r = 5..9 (i=1 block): U[0*m+j, r] = 0, U[1*m+j, r] = U1[j, r-5]
    
    # Similarly for V and W.
    
    # For 1x5x5: C[k] = sum_j A[j] B[j,k]
    # U0[j, r] = 1 for all j, r? No, that doesn't work.
    # Schoolbook for 1x5x5:
    # For each (j, k): U[j, col] = 1, V[j*p+k, col] = 1, W[k, col] = 1
    # So U0 is 5x5 identity-like, V0 is 25x5, W0 is 5x5.
    
    # Let me just implement the block decomposition with rank 10.
    
    def build_block_decomposition():
        R = 10
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # For each i in {0, 1}, use a 1x5x5 schoolbook decomposition
        for i in range(n):
            base_r = i * 5  # Offset in the rank dimension
            for j in range(m):
                for k in range(p):
                    r = base_r + (j * p + k) % 5  # This is wrong, let me redo
                    
        # Let me redo this more carefully.
        # For 1x5x5, schoolbook uses 25 terms (one for each (j,k) pair).
        # But we want to use rank 5 for 1x5x5.
        # Actually, 1x5x5 schoolbook rank is 5*5 = 25, not 5.
        # The rank of the 1xn x nxn tensor is at least n, but schoolbook gives n^2.
        # Can we achieve rank n for 1xn x nxn?
        # C[k] = sum_j A[j] B[j,k]
        # We need: sum_r U[j,r] V[j,k,r] W[k,r] = 1 for all j,k.
        # If we set U[j,r] = delta_{j,r}, V[j,k,r] = delta_{j,r}, W[k,r] = 1,
        # then sum_r delta_{j,r} delta_{j,r} * 1 = 1 if j is in range, which works.
        # But V[j,k,r] = delta_{j,r} means V has shape (m*p, R) = (25, 5).
        # V[j*p+k, r] = delta_{j,r} for r in 0..4.
        # This works! Rank 5 for 1x5x5.
        
        # So for 2x5x5, we can use rank 2*5 = 10.
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for i in range(n):
            base_r = i * m  # 0 for i=0, 5 for i=1
            for j in range(m):
                for k in range(p):
                    r = base_r + j
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
        
        return U, V, W
    
    U, V, W = build_block_decomposition()
    
    # Verify
    def is_valid(U, V, W, tol=1e-8, seed=42):
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        R = U.shape[1]
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        L = U @ A_flat
        M = V @ B_flat
        S = L * M
        
        W_reshaped = W.T.reshape(R, p, n)
        C_rec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_rec[i, k] = np.dot(S, W_reshaped[:, k, i])
        
        return np.allclose(C, C_rec, atol=tol)
    
    if is_valid(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If the block decomposition doesn't work, fall back to schoolbook
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
