"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    # Reshape to (rank, n, m) for U, (rank, m, p) for V, (rank, p, n) for W
    U_reshaped = U_arr.reshape(rank, n, m)
    V_reshaped = V_arr.reshape(rank, m, p)
    W_reshaped = W_arr.reshape(rank, p, n)
    
    T = np.einsum('rab,rcd,rki->abcdik', U_reshaped, V_reshaped, W_reshaped)
    
    # Check non-zero entries
    for a in range(n):
        for b in range(m):
            for k in range(p):
                if abs(T[a, b, b, k, a, k] - 1.0) > tol:
                    return False
    
    # Check zero entries
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            if abs(T[a, b, c, d, i, k] - expected) > tol:
                                return False
    
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _make_strassen_2x4x4():
    """
    Create a rank-7 decomposition for multiplying a 2x4 matrix by a 4x4 matrix
    to produce a 2x4 matrix, using Strassen's 4x4x4 algorithm with zero-padding.
    
    We pad the 2x4 A to 4x4 (add 2 zero rows), use Strassen rank-7 for 4x4x4,
    and take the first 2 rows of the 4x4 result.
    
    Returns U (2*4 x 7), V (4*4 x 7), W (4*2 x 7) for the sub-problem:
    A_sub: 2x4 (indices a=0,1; b=0..3)
    B_sub: 4x4 (indices c=0..3; d=0..3)
    C_sub: 2x4 (indices i=0,1; k=0..3)
    
    In the full 2x5x5 tensor:
    - U entries: U[a*m+b, r] where a in {0,1}, b in {0,1,2,3}
    - V entries: V[c*p+d, r] where c in {0,1,2,3}, d in {0,1,2,3}
    - W entries: W[k*n+i, r] where k in {0,1,2,3}, i in {0,1}
    """
    n_full, m_full, p_full = 2, 5, 5
    
    # Strassen's 7 products for 4x4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, split into 2x2 blocks. Each 2x2 block has 4 entries.
    # A is 4x4, B is 4x4, C is 4x4.
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
    #            A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # Similarly for B and C.
    
    # But our A is only 2x4, padded to 4x4 with 2 zero rows.
    # So A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = zeros, A22 = zeros.
    # B is 4x4: B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C is 4x4: C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    # We only care about C11 and C12 (first 2 rows).
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    
    # Since A21 = A22 = 0:
    # M1 = (A11 + A22)(B11 + B22) = A11(B11 + B22)
    # M2 = (A21 + A22)B11 = 0
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11) = 0
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12) = -A11(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22) = A12(B21 + B22)
    
    # So:
    # C11 = M1 + 0 - M5 + M7 = A11(B11+B22) - (A11+A12)B22 + A12(B21+B22)
    #      = A11*B11 + A11*B22 - A11*B22 - A12*B22 + A12*B21 + A12*B22
    #      = A11*B11 + A12*B21
    # C12 = M3 + M5 = A11(B12-B22) + (A11+A12)B22 = A11*B12 - A11*B22 + A11*B22 + A12*B22 = A11*B12 + A12*B22
    
    # So C11 = A11*B11 + A12*B21 and C12 = A11*B12 + A12*B22
    # This is just the schoolbook for the 2x4 times 4x4 case!
    # That gives 4*4 = 16 terms, not 7.
    
    # The issue: Strassen's reduction to 7 only works when all 4 blocks of A are non-trivial.
    # With A21=A22=0, we lose the savings.
    
    # Let me reconsider. Maybe I should use a different embedding strategy.
    # 
    # Alternative: Use the 2x5x5 problem directly with a known low-rank decomposition.
    # 
    # Actually, let me think about this differently. The tensor for 2x5x5 matrix multiplication
    # has a known rank. Let me try a different approach: use bilinear complexity bounds.
    #
    # For now, let me just return schoolbook since the Strassen embedding doesn't help
    # for rectangular matrices with zero-padded rows.
    
    # Actually, wait. Let me reconsider the problem structure.
    # 
    # The key insight: for C[i,k] = sum_j A[i,j] * B[j,k], we need n*p output entries,
    # each requiring m scalar products. Schoolbook = n*m*p.
    #
    # Can we share computations? If two output entries C[i,k1] and C[i,k2] share the same
    # A[i,j] factors, we might combine them.
    #
    # For fixed i, C[i,0:5] = A[i,0:5] @ B[0:5, 0:5]. This is a 1x5 times 5x5 = 1x5.
    # The rank of a 1xm x mxp multiplication is at most min(m*p, ...). Actually for 1xm x mxp,
    # the bilinear complexity is exactly m*p (you need all m*p products) unless you can
    # exploit structure.
    #
    # Hmm, but actually for 1x5 x 5x5, the rank is 25 (schoolbook). No savings there.
    #
    # Let me try yet another approach: use the fact that we can group terms.
    # 
    # Actually, I realize the Strassen approach doesn't directly apply because of the
    # rectangular nature. Let me just use schoolbook and try a smarter search.
    
    # Return schoolbook as fallback
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try structured rank reduction approaches
    # For 2x5x5, try to find decompositions below 50
    
    # Approach 1: Try rank 25 (half of schoolbook) with structured initialization
    # The idea: for each row i of A, we compute row i of C = A[i,:] @ B
    # This is a 1x5 * 5x5 = 1x5 problem, which has rank 25.
    # But maybe we can do better by sharing across rows.
    
    # Let's try a gradient-based search for various target ranks
    np.random.seed(42)
    
    for target_rank in [25, 30, 35, 40, 45]:
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 20:
            break
        
        budget = min(remaining * 0.4, 15.0)
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Build target tensor
        T_target = np.zeros((n, m, m, p, n, p))
        for a in range(n):
            for b in range(m):
                for k in range(p):
                    T_target[a, b, b, k, a, k] = 1.0
        
        def compute_tensor(U, V, W):
            U_r = U.T.reshape(target_rank, n, m)
            V_r = V.T.reshape(target_rank, m, p)
            W_r = W.T.reshape(target_rank, p, n)
            return np.einsum('rab,rcd,rki->abcdik', U_r, V_r, W_r)
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(3000):
            if not still_going():
                break
            
            T_current = compute_tensor(U, V, W)
            diff = T_current - T_target
            current_loss = np.sum(diff**2)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            U_r = U.T.reshape(target_rank, n, m)
            V_r = V.T.reshape(target_rank, m, p)
            W_r = W.T.reshape(target_rank, p, n)
            
            grad_U = 2 * np.einsum('abcdik,rcd,rki->rab', diff, V_r, W_r)
            grad_V = 2 * np.einsum('abcdik,rab,rki->rcd', diff, U_r, W_r)
            grad_W = 2 * np.einsum('abcdik,rab,rcd->rki', diff, U_r, V_r)
            
            grad_U = grad_U.reshape(target_rank, n*m).T
            grad_V = grad_V.reshape(target_rank, m*p).T
            grad_W = grad_W.reshape(target_rank, p*n).T
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            lr *= 0.9999
        
        if best_loss < 1e-10:
            if _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
                best = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
