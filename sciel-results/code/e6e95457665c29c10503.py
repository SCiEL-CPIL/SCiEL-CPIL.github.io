"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n + i, r] convention
    
    def elapsed_ok():
        return time.time() - start_time < deadline
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify exact decomposition using Fraction arithmetic."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Convert to Fraction for exactness
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        # For each output element C[i,k], check that:
        # sum_j A[i,j]*B[j,k] = sum_r U[i*m+j... wait, let me re-read the convention
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But matrix mult tensor is T[i,j,k] = delta_{j,k}... no.
        # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] for matmul has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # Wait, the standard matmul tensor: T_{i,j,k} where i indexes A row, j indexes A col/B row, k indexes B col
        # T[i,j,k] = 1 (it's the identity in the middle index)
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # But here U is flattened: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is matmul
        # Actually the matmul tensor T[i,j,k] = 1 for all i,j,k? No.
        # For matrix multiplication, the tensor is T[i,j,k] = δ_{j,k}... no that's not right either.
        
        # Let me reconsider. The matrix multiplication tensor T ∈ R^{n×m×p} where
        # T[i,j,k] represents the coefficient such that C[i,k] = sum_{j} A[i,j] B[j,k]
        # This means T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes with coefficient 1)
        # Wait no. The tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give contribution to C[i,k].
        # Since C[i,k] = sum_j A[i,j] B[j,k], we have T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Hmm, that doesn't seem right. Let me think again.
        
        # Actually, the standard way: the matmul tensor T has T[i,j,k] = 1 for all i,j,k.
        # No wait. Let's be precise. We have A ∈ R^{n×m}, B ∈ R^{m×p}, C = AB ∈ R^{n×p}.
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
        # The tensor T ∈ R^{n×m×p} is defined by T[i,j,k] = 1 for all i,j,k? 
        # That can't be because then the rank would be n*m*p which matches schoolbook.
        
        # Yes! The matmul tensor T[i,j,k] = 1 for all i,j,k. The rank of this all-ones tensor... 
        # no that's not right either. Let me look at this differently.
        
        # The tensor for matrix multiplication: T[i,j,k] where the trilinear form is
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[k,i] (with some dual)
        # For matmul, T[i,j,k] = δ_{jk}... no.
        
        # I think the correct formulation: The matmul tensor has entries T[i,j,k] = 1 
        # when we're pairing A[i,j] with B[j,k] to produce C[i,k]. Since every such pairing 
        # exists, T[i,j,k] = 1 for all i,j,k. But that gives an all-ones tensor which has rank...
        
        # Actually no. Let me reconsider. The tensor T ∈ R^{n×m×p} for multiplying 
        # n×m by m×p matrices is defined such that:
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # The tensor T satisfies: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
        
        # I think the standard definition is T[i,j,k] = 1 for all i,j,k, and the 
        # decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] must hold.
        # With the flattening: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using numpy (may have floating point issues, use for initial screening)."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ≈ 1
        # This is equivalent to checking that the tensor T[i,j,k] = sum_r ... equals 1
        
        # Build the tensor implicitly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(Ua[i*m+j], Va[j*p+k]) * 0  # placeholder
                    # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # = sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r]
                    # Let's just compute directly
                    vals = Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                    if abs(np.sum(vals) - 1.0) > 1e-6:
                        return False
        return True
    
    def try_random_search(max_rank, n_trials=100):
        """Try to find a decomposition of given rank using random init + local search."""
        n, m, p = DIMENSIONS
        best = None
        
        for trial in range(n_trials):
            if not elapsed_ok():
                break
            
            U = np.random.randn(n*m, max_rank)
            V = np.random.randn(m*p, max_rank)
            W = np.random.randn(p*n, max_rank)
            
            # Simple gradient-free local search: perturb and accept if closer to target
            # Target: T[i,j,k] = 1 for all i,j,k
            # Error: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            
            def error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(max_rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            err += (s - 1.0)**2
                return err
            
            current_err = error(U, V, W)
            
            # Local search with decreasing step size
            step = 1.0
            for it in range(200):
                if not elapsed_ok():
                    break
                
                # Random perturbation
                dU = np.random.randn(n*m, max_rank) * step
                dV = np.random.randn(m*p, max_rank) * step
                dW = np.random.randn(p*n, max_rank) * step
                
                new_U = U + dU
                new_V = V + dV
                new_W = W + dW
                new_err = error(new_U, new_V, new_W)
                
                if new_err < current_err:
                    U, V, W = new_U, new_V, new_W
                    current_err = new_err
                    step = min(step * 1.1, 5.0)
                else:
                    step *= 0.95
                
                if current_err < 1e-10:
                    break
            
            if current_err < 1e-8:
                # Try to convert to rational
                # Round to nearest half-integer or integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = [[int(x) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in U_rounded]
                V_list = [[int(x) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in V_rounded]
                W_list = [[int(x) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in W_rounded]
                
                if verify_fast(U_list, V_list, W_list):
                    if verify(U_list, V_list, W_list):
                        if best is None or max_rank < best[0]:
                            best = (max_rank, U_list, V_list, W_list)
        
        return best
    
    # Try to find lower rank decompositions
    # Start from schoolbook rank 48 and try to go lower
    # Known: for small matrices, Strassen-like methods can help
    # For 2x3 x 3x8, let's try ranks from 47 down
    
    best_result = schoolbook()
    best_rank = n * m * p  # 48
    
    # Try various ranks
    for rank in range(47, 20, -1):
        if not elapsed_ok():
            break
        
        result = try_random_search(rank, n_trials=5)
        if result is not None:
            best_rank = rank
            best_result = {"U": result[1], "V": result[2], "W": result[3]}
            break
    
    return best_result
# EVOLVE-BLOCK-END
