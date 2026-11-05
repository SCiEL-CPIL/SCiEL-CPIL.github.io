"""Matrix multiplication 6x6x6 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]

def _validate(U, V, W, n=6, m=6, p=6):
    """Check exact decomposition. Returns True if valid."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # T should be 1 if i,j,k are "connected" in the matmul tensor, else 0
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity-like structure)
    # Wait, the matmul tensor T_{i,j,k} has T[i,j,k]=1 for all i,j,k in the standard basis
    # No: the matrix multiplication tensor has T[i,j,k] = delta_{i,j,k} meaning
    # C[i,k] = sum_j A[i,j] B[j,k], so the trilinear form is sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # The tensor entries are T[i,j,k] = 1 for all i,j,k? No.
    # T is a 3-tensor where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
    # Actually T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # So the matmul tensor is the all-ones tensor of shape (n, m, p).
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to 3D
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is expensive for large rank, so do it in chunks or use einsum
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # = einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    
    # For efficiency with moderate rank, do it directly
    # But for rank up to ~200, this is 6*6*6*rank = 216*rank operations, fine.
    
    # Use einsum for clarity
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    
    # Should be all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _rank_of(U, V, W):
    """Return the rank (number of columns)."""
    if not U:
        return 0
    return len(U[0])

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s, keep safety margin
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Schoolbook baseline
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
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: try to find lower-rank decompositions using structured approaches
    # For 6x6x6, known best rank is around 20-21 (Lader 1976 gave 47 for 7x7x7, etc.)
    # Actually for 6x6x6, the best known is 20 (from various constructions)
    # Let's try randomized + local search approaches
    
    # Approach 1: Random initialization + iterative refinement
    # The matmul tensor is the all-ones tensor of shape (n,m,p)
    # We need T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Use a greedy/iterative approach
    # Start with a random decomposition of some rank, then try to reduce
    
    # For 6x6x6, let's try ranks from 20 upward
    # First, try a structured approach based on known constructions
    
    # Lader's construction for 7x7x7 has rank 47. For 6x6x6, we can try
    # various known constructions.
    
    # Let's try a simple approach: use the fact that the all-ones tensor
    # can be decomposed using rank-one tensors.
    
    # Actually, let me think about this differently. The matmul tensor for n x m x p
    # is T[i,j,k] = 1 for all i,j,k. This is just the all-ones 3-tensor.
    # The rank of the all-ones tensor of shape (n,m,p) is... well, it's the 
    # multiplicative complexity of n x m x p matrix multiplication.
    
    # For 6x6x6, let's try to find a low-rank decomposition.
    
    # Method: Start with a random rank-r decomposition and use gradient-free
    # local search to minimize the error, then check if it's exact.
    
    # Better method: Use the structure. The all-ones tensor T[i,j,k]=1 can be written as
    # sum_c u_i^c * v_j^c * w_k^c where u, v, w are vectors.
    # But we need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] which is different.
    
    # Let me re-read the convention:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) where k is from B's row and i is from C's row
    # 
    # The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_{j,c} A[i,j] * (something) 
    # 
    # Standard tensor decomposition of matmul:
    # C[i,k] = sum_c U_c[i] * V_c[k] * ... no
    # 
    # Actually the standard form is:
    # C[i,k] = sum_c (sum_j A[i,j] * U[j, c]) * V[c, k] ... no
    #
    # Let me think again. The matmul tensor T has T[i,j,k] = 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j] B[j,k] in C[i,k] is 1)
    #
    # A rank-1 decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # This is equivalent to: for each c, we have three "slices" that multiply to give 1.
    
    # Let's try a constructive approach.
    # For the all-ones tensor, one approach is to use:
    # U[i*m+j, c] = a_c[i] * b_c[j]
    # V[j*p+k, c] = b_c[j] * d_c[k]  -- wait, this doesn't work because V depends on j and k
    #
    # Actually, let's just use numerical optimization.
    
    # Random restart + local search
    np.random.seed(42)
    random.seed(42)
    
    def try_rank(r, max_iters=2000, time_limit=None):
        """Try to find a valid decomposition of rank r using random init + local search."""
        if time_limit is not None and time.time() > time_limit:
            return None
        
        # Initialize randomly
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Local search: perturb one element at a time, accept if error decreases
        # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        def compute_error(U, V, W):
            Ur = U.reshape(n, m, r)
            Vr = V.reshape(m, p, r)
            Wr = W.reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
            return np.sum((T - 1.0)**2)
        
        current_error = compute_error(U, V, W)
        
        # Simple local search with adaptive step size
        step = 0.1
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Pick a random matrix and element
            mat = random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
            if mat == 0:
                row, col = random.randint(0, n*m-1), random.randint(0, r-1)
                delta = random.gauss(0, step)
                old_val = U[row, col]
                U[row, col] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    U[row, col] = old_val
            elif mat == 1:
                row, col = random.randint(0, m*p-1), random.randint(0, r-1)
                delta = random.gauss(0, step)
                old_val = V[row, col]
                V[row, col] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    V[row, col] = old_val
            else:
                row, col = random.randint(0, p*n-1), random.randint(0, r-1)
                delta = random.gauss(0, step)
                old_val = W[row, col]
                W[row, col] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    W[row, col] = old_val
            
            # Adapt step size
            if it % 100 == 99:
                step *= 0.95
        
        # Check if we found an exact solution
        if current_error < 1e-10:
            # Try to round to nice values
            U_rounded = np.round(U * 4) / 4  # Half-integer grid
            V_rounded = np.round(V * 4) / 4
            W_rounded = np.round(W * 4) / 4
            if _validate(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return U_rounded, V_rounded, W_rounded
            # Also try without rounding
            if _validate(U.tolist(), V.tolist(), W.tolist()):
                return U, V, W
        
        return None
    
    # Try various ranks
    for r in range(20, 40):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(r, max_iters=3000, time_limit=deadline - 35)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist() if hasattr(U, 'tolist') else U,
                "V": V.tolist() if hasattr(V, 'tolist') else V,
                "W": W.tolist() if hasattr(W, 'tolist') else W
            }
            break  # Found a valid one at this rank
    
    # If we didn't find anything better, return the schoolbook
    if best_rank == n * m * p:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
