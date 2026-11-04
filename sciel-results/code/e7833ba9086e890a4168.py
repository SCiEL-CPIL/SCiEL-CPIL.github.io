"""Program for matrix multiplication 2x3x7 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    # Start with schoolbook decomposition (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    # Try to find lower rank decompositions using random projections + verification
    # Strategy: Try various ranks below 42, use random initialization + refinement
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The multiplication tensor M[i,j,k] should equal sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # U has shape (nm, rank), V has shape (mp, rank), W has shape (pn, rank)
        # For output (i, k): M[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i*m+j, j*p+k, k*n+i] = delta (it's 1 for all valid indices)
        # Decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # So T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        rank = U.shape[1]
        # Compute the decomposed tensor
        # This is expensive, so we do it efficiently
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # For efficiency, compute for all (a,b,c) where a in [0,nm), b in [0,mp), c in [0,pn)
        # But we only need to check the structure of matrix multiplication
        
        # The matrix multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 
        # and T[a,b,c] = 0 otherwise
        
        # Let's verify by checking random samples and key structural properties
        # Actually, let's do a proper check using matrix operations
        
        # For each rank component r, the contribution to output[i,k] is:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But W depends on (k,i) so this is tricky.
        
        # Better approach: The tensor T has entries T[a,b,c] where:
        # a = i*m + j (input A index)
        # b = j'*p + k (input B index) 
        # c = k'*n + i' (output index)
        # T[a,b,c] = 1 if j=j' and i=i' and k=k', else 0
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1
        
        # For the decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Let's verify by computing for all valid (i,j,k) triples
        # and checking that T[i*m+j, j*p+k, k*n+i] = 1
        # and for invalid combinations, T = 0
        
        # Efficient verification:
        # Compute all outputs and check
        
        # For each (i, k) pair in output, the value should be sum_j A[i,j]*B[j,k]
        # In the tensor decomposition, output[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        
        # Let's build the implied multiplication and check it matches
        
        # Actually, the cleanest way: 
        # The tensor T has rank at most R if there exist U, V, W such that
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # Let's verify by sampling
        np.random.seed(42)
        
        # Check 1: For all (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check 2: For random invalid (a,b,c), T should be 0
        # Invalid means not of the form (i*m+j, j*p+k, k*n+i)
        for _ in range(2000):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a valid multiplication term
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    # Try to find lower rank using iterative refinement
    # Use a random start and then try to zero out components
    
    # Try ranks from 41 down to some minimum
    # Theoretical lower bound: rank >= n*m + m*p + p*n - n - m - p + 1 (not tight)
    # For 2x3x7: at least... let's just try various ranks
    
    for target_rank in range(41, 20, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            
            # Random initialization
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Use gradient-free optimization: try to minimize error
            # Error = ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||
            
            # Since we need exact decomposition, try a different approach:
            # Start from schoolbook and try to merge/eliminate components
            
            # Alternative: Use the fact that we can try to find a basis change
            # that makes some components zero
            
            # Let's try: start from schoolbook, project to lower rank
            # and see if we can recover exactness
            
            U_proj = np.random.randn(target_rank, rank_sb)
            V_proj = np.random.randn(target_rank, rank_sb)
            W_proj = np.random.randn(target_rank, rank_sb)
            
            U_new = U_sb @ U_proj.T  # (nm, target_rank)
            V_new = V_sb @ V_proj.T  # (mp, target_rank)  
            W_new = W_sb @ W_proj.T  # (pn, target_rank)
            
            if verify_decomposition(U_new, V_new, W_new):
                best_rank = target_rank
                best_U = U_new.tolist()
                best_V = V_new.tolist()
                best_W = W_new.tolist()
                break
        
        if best_rank < 42:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == 42:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
