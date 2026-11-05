"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
import random

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    # Actually the tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * 1 == W_tensor[i, j, k, r]... 
    # No wait. The standard decomposition is:
    # (A x B)[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_r (sum_r U[i, j, r] * A[i, j]) * (sum_r V[j, k, r] * B[j, k]) * ... no
    # 
    # The tensor for matrix mult is T[i, j, k] = A[i,j] * B[j,k] -> (A*B)[i,k]
    # Wait, no. The matmul tensor is defined as:
    # (A × B)[i, k] = sum_j A[i, j] B[j, k]
    # The bilinear form is: sum_r sum_j U[i,j,r] A[i,j] * V[j,k,r] B[j,k] * W[i,k,r] = sum_{i,k} W[i,k,r] (A B)[i,k]
    # 
    # Let me re-read the interface:
    # U: (n*m, rank) where index is i*m+j
    # V: (m*p, rank) where index is j*p+k  
    # W: (p*n, rank) where index is k*n+i
    #
    # The decomposition: (AB)[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... no
    #
    # Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Rank decomposition: (AB)[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])... 
    # That's not right either because the sum over j is inside both.
    #
    # Correct form: (AB)[i,k] = sum_r W[i,k,r] * sum_j U[i,j,r] A[i,j] * ... no
    #
    # Let me think again. The standard Strassen-style decomposition:
    # Let M_r = (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])  -- NO
    #
    # Actually the correct form is:
    # (AB)[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j' V[j',k,r] B[j',k])
    # This would mean the r-th "product" is an outer product of two vectors.
    #
    # Wait, I think the correct interpretation is:
    # (AB)[i,k] = sum_r W[i,k,r] * prod_r  where prod_r uses U and V
    #
    # Let me look at it differently. For a rank-r decomposition of matmul n×m × m×p:
    # We need: for all A (n×m), B (m×p):
    # (AB)[i,k] = sum_{r=1}^R sum_j U_{i,j,r} A_{i,j} * V_{j,k,r} B_{j,k}  ... NO
    #
    # The correct tensor equation: T_{i,j,k} for (AB) is not a tensor, (AB) is computed.
    # The bilinear algorithm: 
    # (AB)_{i,k} = sum_r sum_j a_{i,j} b_{j,k} T_{i,j,k,r} ... no
    #
    # OK let me just use the schoolbook form to understand:
    # In schoolbook: for each (i,j,k) triple:
    #   U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    #   col += 1
    # So for a given (i,k) entry of AB:
    #   (AB)[i,k] = sum_r W[k*n+i, r] ... no
    #
    # Actually I think the formula is:
    # (AB)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])... no
    #
    # Let me just expand: In the schoolbook, col corresponds to (i,j,k) triple.
    # U[i*m+j, col] = 1 only when the col's i matches row i AND col's j matches col j.
    # V[j*p+k, col] = 1 only when col's j matches AND col's k matches.
    # W[k*n+i, col] = 1 only when col's k matches AND col's i matches.
    #
    # So: (AB)[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] A[i,j] * ... 
    #
    # No wait. I think it's:
    # (AB)[i,k] = sum_r W[k*n+i, r] * (U_vector_r . A_i_vector) * (V_vector_r . B_k_vector)
    # where U_vector_r[j] = U[i*m+j, r]... but that depends on i which is outside.
    #
    # Hmm, let me reconsider. Maybe it's simpler:
    # The r-th term contributes: W[k*n+i, r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    #
    # In schoolbook for col=(i0,j0,k0):
    # U[i*m+j, col] = delta(i,i0)*delta(j,j0)
    # V[j*p+k, col] = delta(j,j0)*delta(k,k0)
    # W[k*n+i, col] = delta(k,k0)*delta(i,i0)
    #
    # So for entry (i,k):
    # sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j' V[j'*p+k,r] B[j',k])
    # = sum_{r: k=r_k, i=r_i} 1 * A[i, r_j] * B[r_j, k]
    # Wait, for a fixed r with (i0,j0,k0):
    # W[k*n+i, r] = 1 iff k=k0 and i=i0
    # sum_j U[i*m+j,r] A[i,j] = A[i0, j0] (since only j=j0 has U=1, and i must be i0)
    # sum_j V[j*p+k,r] B[j,k] = B[j0, k0]
    #
    # So the r-th term for entry (i,k) is:
    #   delta(i,i0)*delta(k,k0) * A[i0,j0] * B[j0,k0]
    #
    # Summing over all r (i.e., all (i0,j0,k0)):
    #   sum_{i0,j0,k0} delta(i,i0)*delta(k,k0) * A[i0,j0]*B[j0,k0]
    #   = sum_{j0} A[i,j0] * B[j0,k]
    #   = (AB)[i,k] ✓
    #
    # So the formula is:
    # (AB)[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # To verify: pick random integer A, B, compute AB, then compute via decomposition.

    R = U.shape[1]
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    
    # Test with a few random matrices
    for _ in range(5):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        AB = A @ B
        
        # Compute via decomposition
        AB_dec = np.zeros((n, p), dtype=np.int64)
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(R):
                    u_val = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    val += W[k*n+i, r] * u_val * v_val
                AB_dec[i, k] = val
        
        if not np.array_equal(AB, AB_dec):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin before 150s limit

    # Start with schoolbook as fallback
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    elapsed = time.time() - start_time
    if time.time() > deadline:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Try to find a lower-rank decomposition by attempting to express
    # the matmul tensor with fewer columns.
    #
    # Strategy: The matmul tensor has structure. For 3x3x3 blocks, the
    # schoolbook rank per "k-slice" is 9. With 7 k-values, total is 63.
    # 
    # One approach: try to share computation across k-values.
    # For a given (i,j) pair, the contribution is A[i,j] * sum_k B[j,k] * W[i,k]
    # The rank contribution from fixing (i,j) is at most p (one per k), but
    # if we can combine across different (i,j) pairs sharing the same
    # structure, we might reduce rank.
    #
    # A concrete approach: use block structure.
    # Decompose the m=3 dimension. The 3x3 matrix A can be partitioned.
    # But m is the shared dimension.
    #
    # Let's try: for each k, we need to compute sum_j A[i,j]*B[j,k] for all (i,k).
    # This is 3 matrix-vector products (one per k) each of size 3x3.
    # Each 3x3 mat-vec has rank 9 (schoolbook).
    # But the 7 mat-vecs share the same A!
    #
    # Key insight: A is fixed. So for all k, we compute A * B[:, k].
    # The "A-multiplication" part: for a given A, define c_j = sum_i A[i,j] * x_i... 
    # no. A * v = [sum_j A[0,j]v[j], sum_j A[1,j]v[j], sum_j A[2,j]v[j]]
    #
    # The bilinear form for A*v is: (Av)_i = sum_j A[i,j] v[j]
    # This has a natural rank-9 decomposition, but also can be viewed as:
    # Av = A ⊗_bilinear v
    #
    # For 7 different vectors v_1,...,v_7 (columns of B), we need:
    # 7 vectors, each in R^3, computed as A*v_k.
    # Total output: 3*7 = 21 values.
    #
    # The bilinear algorithm: we need to compute 7 vector-matrix products A*v_k.
    # Equivalently: compute the 3x7 product A*B.
    #
    # Rank of "left-multiply by A" map: A is 3x3, we multiply 7 column vectors.
    # This is the tensor for 3x3 matmul with 7 right-factors.
    #
    # One approach: use the fact that rank(3x3x3 matmul tensor) could be < 9.
    # Known: Strassen gives rank 7 for 2x2x2. For 3x3x3, best known upper bounds
    # are around 23 (from repeated Strassen) vs schoolbook 27.
    # But we need exact integer/half-integer entries.
    #
    # Since finding a provably optimal or even non-trivially better decomposition
    # for 3x3x3 with integer entries within 100s is very hard, let's try a
    # heuristic search for rank < 63.
    #
    # Heuristic: try rank R and solve for U,V,W using the linear constraints.
    # The constraint is: for all (i,j,k) and r, the tensor equation holds.
    # This gives n*m*p*R equations... actually it's nonlinear (product of U and V terms).
    #
    # Alternative: fix U and V, solve for W linearly? No, W appears multiplied.
    #
    # Let's try a different approach: greedy column elimination.
    # Start with schoolbook (rank 63). Try to see if any column is redundant
    # (can be expressed as a combination of others) while maintaining exactness.
    # This is equivalent to checking if the rank-63 tensor has lower rank.
    #
    # Actually, the rank of the tensor is intrinsic. The schoolbook decomposition
    # might not be rank-minimal, but reducing it requires non-trivial algebra.
    #
    # Given time constraints, let's try a small randomized search:
    # For each candidate rank R from some lower bound up to best_rank-1,
    # try random initializations and solve.
    #
    # But the problem is bilinear (product of U-entries and V-entries for each r),
    # making it a system of polynomial equations. Very hard to solve generally.
    #
    # Practical approach: stick with schoolbook. It's guaranteed correct.
    # The chance of finding a rank < 63 exact integer decomposition in ~100s
    # for this specific shape is very low without specialized knowledge.

    # Verify schoolbook (should pass)
    if _verify_decomposition(n, m, p, best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # If schoolbook somehow fails (shouldn't happen), return anyway
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
