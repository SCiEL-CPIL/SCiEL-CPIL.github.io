"""Rewritten program: exploit rectangular geometry 2x3x10 via 2x2x10 + 1x1x10 with WFTA on 2x2."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def make_tensor(n, m, p):
        """Create the matrix multiplication tensor indexed by (i*m+j, j*p+k, k*n+i)."""
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T

    def verify(U, V, W, T, tol=1e-8):
        """Check if U, V, W decompose T."""
        rank = U.shape[1]
        result = np.zeros(T.shape, dtype=np.float64)
        for c in range(rank):
            result += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return np.max(np.abs(result - T)) < tol

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    T = make_tensor(n, m, p)

    # Schoolbook fallback: rank = n*m*p = 60
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    best_rank = rank_schoolbook

    # Strategy: decompose 2x3x10 into 2x2x10 (rows 0,1) + 1x1x10 (row 2)
    # The 2x2x10 sub-problem: multiply 2x2 by 2x10
    # The 1x1x10 sub-problem: multiply 1x1 by 1x10 (trivial, rank 10)
    # Total rank if we can do 2x2x10 with rank 30: 30 + 10 = 40
    # But we want to try WFTA to reduce 2x2x10 further.
    
    # For 2x2x10: the tensor has structure. Let's think about it differently.
    # Actually, let's try a direct approach: use the fact that for 2x2x10,
    # we can use Strassen for the 2x2 part giving rank 7 per "column" of the 10,
    # but that gives 7*10 = 70 which is worse than schoolbook 40.
    # 
    # Better: For 2x2x10, schoolbook is 2*2*10 = 40.
    # For 1x1x10, schoolbook is 1*1*10 = 10.
    # Total schoolbook: 50, but the full 2x3x10 schoolbook is 60.
    # The difference is because the 2x3x10 has 3*2*10 = 60 terms.
    # 
    # Wait, let me reconsider. The full problem is 2x3 times 3x10.
    # We can split the 3 middle dimension: first 2 columns and last 1 column.
    # So A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x10) = [B1 (2x10); B2 (1x10)]
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2 * 2x10, rank 40 schoolbook
    # A2*B2 is 2x1 * 1x10, rank 20 schoolbook
    # Total: 60
    #
    # Alternative split: first 1 column and last 2 columns:
    # A (2x3) = [A1 (2x1) | A2 (2x2)]
    # B (3x10) = [B1 (1x10); B2 (2x10)]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1 * 1x10, rank 20
    # A2*B2: 2x2 * 2x10, rank 40
    # Total: 60
    #
    # So we need to find a better decomposition for 2x2x10.
    # The minimal rank for 2x2x10 matrix multiplication...
    # For 2x2xn, the rank is known to be 2n + O(1) for large n, but for small n it varies.
    # For 2x2x10, let's try to find a low-rank decomposition.
    
    # Let's try to find a decomposition for 2x2x10 with rank less than 40.
    # We'll use the structure: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,9}
    
    # Try rank 30 for 2x2x10 (if possible)
    # Actually, let's just try to optimize for the full problem directly with various ranks.
    
    # Let's try a structured approach: use the WFTA idea for the 2x2 part.
    # For 2x2x10, we can view it as 10 independent 2x2x1 multiplications? No, that's not right.
    # Actually, 2x2x10 means multiplying a 2x2 matrix by a 2x10 matrix.
    # This is equivalent to 10 independent 2x2x1 multiplications (each column of the result).
    # For 2x2x1, the rank is 4 (schoolbook) or 7 (Strassen, but Strassen is for 2x2x2).
    # For 2x2x1, the minimal rank is 4.
    # So 2x2x10 has rank at most 40 (schoolbook) and at least... well, it's at least 2*10 = 20 (since each of the 10 outputs needs at least 2 terms? No, that's not a valid lower bound).
    
    # Let's just try to find a good decomposition using optimization.
    # We'll try ranks from 35 up to 60 and see if we can find one.
    
    for target_rank in range(35, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try a few different initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + trial * 17)
            
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Use a simple iterative refinement
            # We'll use a projected gradient approach
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                R = np.zeros(T.shape, dtype=np.float64)
                for c in range(target_rank):
                    R += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                R -= T
                
                res_norm = np.sqrt(np.sum(R**2))
                if res_norm < 1e-12:
                    found = True
                    break
                
                # Gradient descent: dU[i,c] = sum_{b,d} R[i,b,d] * V[b,c] * W[d,c]
                # This is expensive, so let's use a simpler update
                # Actually, let's use a different approach: alternating least squares
                # For each c, we can solve for U[:,c], V[:,c], W[:,c] given the others
                # But that's also complex.
                
                # Let's just do a simple gradient step
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(target_rank):
                    # dU[:,c] = sum_{b,d} R[:,b,d] * V[b,c] * W[d,c]
                    # We can compute this efficiently
                    tmp = np.einsum('abd,b,c->ad', R, V[:,c], W[:,c])
                    dU[:,c] = tmp
                    dV[:,c] = np.einsum('abd,a,c->bd', R, U[:,c], W[:,c])
                    dW[:,c] = np.einsum('abd,a,b->cd', R, U[:,c], V[:,c])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                break
        
        if found:
            if verify(U, V, W, T, tol=1e-8):
                best_result = (U.copy(), V.copy(), W.copy())
                best_rank = target_rank
                break
    
    U, V, W = best_result
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
