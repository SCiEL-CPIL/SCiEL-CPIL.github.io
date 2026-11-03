"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 21, 21
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: for all i,j,k: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Precompute the tensor as a 3D array
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        result = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c]  # shape (nm,)
            Vc = V[:, c]  # shape (mp,)
            Wc = W[:, c]  # shape (pn,)
            # Reshape and multiply
            Uc3 = Uc.reshape(n, m)
            Vc3 = Vc.reshape(m, p)
            Wc3 = Wc.reshape(p, n)
            # For each c, the contribution is outer product structure
            # Uc3[i,j] * Vc3[j,k] * Wc3[k,i] summed over j,k... no
            # Actually: sum over j,k of Uc3[i,j] * Vc3[j,k] * Wc3[k,i]
            # This is a trace-like operation
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += Uc3[i, j] * Vc3[j, k] * Wc3[k, i]
        return np.allclose(result, T, atol=tol)
    
    def verify_exact(U, V, W):
        """Exact verification using Fractions."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        uc = Fraction(U[i*m+j, c])
                        vc = Fraction(V[j*p+k, c])
                        wc = Fraction(W[k*n+i, c])
                        total += uc * vc * wc
                    if total != 1:
                        return False
        return True
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = 21
    # But actual minimal rank for 3x3x7 is likely higher
    
    # Strategy: For small ranks, use random initialization + alternating minimization
    # For each rank r, try to find U, V, W
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find better decompositions
    # For a given rank r, we can use the following approach:
    # 1. Randomly initialize U and V
    # 2. For each c, the term is a rank-1 tensor in the (i,j,k) space
    # 3. Use alternating least squares or similar
    
    # Actually, let's try a more direct approach for small ranks
    # The key insight: we can fix two matrices and solve for the third
    
    def try_rank(r, max_iters=100, seed=0):
        """Try to find a decomposition of rank r."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        for attempt in range(20):
            # Random initialization
            U = np_rng.randn(nm, r)
            V = np_rng.randn(mp, r)
            W = np_rng.randn(pn, r)
            
            # Alternating minimization
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each k*n+i, we need sum_c V[j*p+k,c] * W[k*n+i,c] * U[i*m+j,c] = 1
                # This is complex due to the coupling. Let's use a different approach.
                
                # Actually, let's use gradient descent on the loss
                # Loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                
                # Compute gradients numerically or analytically
                # This is getting complex. Let's try a simpler approach for small n,m,p.
                
                # For n=3, m=3, p=7, we have 63 equations and 51r unknowns.
                # For r=21, we have 1071 unknowns and 63 equations - very underdetermined.
                # We can set up a linear system if we fix some variables.
                
                break
            
            # Try a different strategy: fix U to be random, then the system in V and W
            # is still nonlinear. 
            
            # Let's try: for each c, the rank-1 tensor is T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We need sum_c T_c = T (all ones)
            
            # This is equivalent to finding r rank-1 tensors that sum to the all-ones tensor.
            # The all-ones tensor of shape (3,3,7) has rank... well, it's not a standard matrix rank.
            
            # Let me try a completely different approach: use the fact that for matrix multiplication,
            # there are known decompositions. For 3x3x3, the minimal rank is 21 (schoolbook is 27,
            # but there's a known rank-21 decomposition). For 3x3x7, we need to extend this.
            
            # Actually, let me just try to solve the system directly for small r.
            # Set up: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # 
            # If we fix U and V, then for each (k,i), the equation becomes:
            # sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
            # Let M[k,i,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then: sum_c M[k,i,c] * W[k*n+i,c] = 1 for all k,i
            # This is a linear system in W!
            
            # So: fix U and V randomly, compute M, solve for W, then check if the full system is satisfied.
            # But the full system also requires the (i,j,k) structure to work out.
            
            # Actually, if we fix U and V, and solve for W from the linear system above, we get W that
            # satisfies: for all k,i: sum_c M[k,i,c] * W[k*n+i,c] = 1
            # But we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # These are different! The linear system I wrote is a projection.
            
            # Let me think again. The full system is 63 equations. If I fix U (9r values) and V (21r values),
            # I have 63 equations in W (21r unknowns). For r >= 1, 21r >= 21 > 63/r... 
            # For r=3, 21r=63, so it's square. For r>3, it's underdetermined.
            
            # So: fix U and V, set up the 63x(21r) system, solve for W, then verify.
            
            U = np_rng.randn(nm, r)
            V = np_rng.randn(mp, r)
            
            # Build the linear system for W
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by (k,i): for fixed k,i, the equation is:
            # sum_c W[k*n+i,c] * (sum over j... no, j is free)
            # 
            # Actually, each (i,j,k) gives a separate equation. The unknowns are W[k*n+i,c] for all k,i,c.
            # For a given (k,i), the unknowns W[k*n+i,c] appear in equations for all j.
            
            # Let's build the full system:
            # Unknowns: W flattened to shape (pn * r,)
            # Equations: for each (i,j,k), one equation
            
            num_eqs = n * m * p
            num_unk = pn * r
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            # W[k*n+i, c] is at index (k*n+i) * r + c
                            w_idx = (k * n + i) * r + c
                            A[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
                        eq_idx += 1
            
            # Solve A @ w_flat = b
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = w_flat.reshape(pn, r)
                
                # Verify
                if verify(U, V, W, tol=1e-7):
                    return U, V, W, r
            except:
                pass
        
        return None
    
    # Try ranks from 21 upward
    for r in range(21, 40):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=10, seed=r)
        if result is not None:
            U, V, W, rank = result
            # Verify exactly
            if verify_exact(U, V, W):
                # Convert to list format
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
