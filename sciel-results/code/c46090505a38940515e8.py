"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    """Construct a tensor decomposition for 3x3x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = 3, 3, 5
    deadline = time.time() + 115  # safety margin for 120s limit
    
    # Start with schoolbook decomposition (rank 45)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for c in range(best_rank):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        best_U[i * m + j][c] = 1
        best_V[j * p + k][c] = 1
        best_W[k * n + i][c] = 1
    
    # Try to find lower rank decompositions
    # Theoretically, rank(3x3x5) could be as low as 19
    # We'll try local search from random initializations
    
    def check_decomposition(U, V, W, rank):
        """Check if decomposition is exact using integer arithmetic."""
        # For each (i, j, k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # and for all other combinations, the sum should be 0
        # Actually, we need: for all i, j, k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_ij * delta_jk * delta_ki
        # Wait, the tensor for matrix multiplication T[i,j,k] = 1 if j=i (first mat row), k=j (second mat col)
        # Actually T[i,j,k] = 1 if the indices form a valid chain: A[i,j]*B[j,k] -> C[i,k]
        # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], which is always true for valid i,j,k
        # The decomposition requires: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[i,k,c]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # But the indexing in the contract says W[k*n+i, r], so W is indexed by (k, i) with k first
        
        # Let me re-read: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The tensor element T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] should be 1 if this corresponds to a valid term in the product
        # Actually, the matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k where the indices are valid
        # No wait - the multiplication tensor satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
        # So T[i,j,k] = 1 if we want to express C[i,k] = sum_j A[i,j] B[j,k]
        # But in the decomposition: sum_c U[i,j,c] V[j,k,c] W[i,k,c] should equal T[i,j,k]
        # Hmm, but W is indexed by (k, i) not (i, k)
        
        # Let me think again. The standard matrix multiplication tensor:
        # T_{i,j,k} where i is row of A/row of C, j is col of A/row of B, k is col of B/col of C
        # T[i,j,k] = 1 if this is a valid term (always for valid indices)
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, W[k*n+i] suggests W is indexed by (k, i)
        
        # The correct identity is: for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) corresponds to a valid multiplication term
        # But actually ALL valid (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid terms
        # So the tensor is all-ones? No...
        
        # Let me reconsider. The matrix multiplication algorithm computes:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form, we can write: C[i,k] * 1 = sum_{j} A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all valid indices, and the identity is:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{something}
        
        # Actually, I think the correct interpretation is:
        # The multiplication tensor M satisfies: M[i,j,k] = 1 for all i,j,k in valid ranges
        # And the decomposition is: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify with schoolbook: for each (i,j,k) triple, we have one rank-1 term
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for the specific c corresponding to (i,j,k)
        # and 0 for others. So M[i,j,k] = 1 for all i,j,k. Yes!
        
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_local_search(rank, max_iter=5000, seed=42):
        """Try to find a decomposition of given rank via local search."""
        np.random.seed(seed)
        
        # Initialize with random small integers
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = np.random.randint(-2, 3, size=(p*n, rank))
        
        # Objective: minimize sum of squared errors
        def objective(U, V, W):
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1) ** 2
            return err
        
        current_obj = objective(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random perturbation
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            
            # Perturb a few entries
            for _ in range(3):
                mat = np.random.choice([U_new, V_new, W_new])
                row = np.random.randint(0, mat.shape[0])
                col = np.random.randint(0, mat.shape[1])
                mat[row, col] += np.random.choice([-1, 1])
            
            new_obj = objective(U_new, V_new, W_new)
            
            if new_obj < current_obj:
                U, V, W = U_new, V_new, W_new
                current_obj = new_obj
                
                if current_obj == 0:
                    # Check exactness with integer arithmetic
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if check_decomposition(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
            elif np.random.random() < 0.1:
                # Accept worse solutions with some probability
                U, V, W = U_new, V_new, W_new
                current_obj = new_obj
        
        return None
    
    # Try ranks from 19 to 44
    for rank in range(19, min(45, best_rank)):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = try_local_search(rank, max_iter=2000, seed=seed + rank * 100)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                break
        
        if best_rank == rank:
            # Found a decomposition at this rank, can stop
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
