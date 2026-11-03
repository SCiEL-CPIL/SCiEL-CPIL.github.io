"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0  # 110s safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But we need to verify: for all i,j,k: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # The multiplication tensor T[i,j,k] = 1 if this is a valid multiplication entry, 0 otherwise
        # Actually, the tensor for matrix multiplication is: T[i,j,k] = 1 for all i,j,k
        # No wait - the tensor encodes: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for the sum)
        # Let me re-derive: we need U, V, W such that:
        # For all i in [0,n), k in [0,p): (A*B)[i,k] = sum_r U[i*m+?, r] * ... 
        # 
        # Standard formulation: T is a tensor in R^{n x m x p} where T[i,j,k] = 1
        # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right either. Let me think more carefully.
        #
        # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T is in R^{(n*m) x (m*p) x (p*n)}? No...
        #
        # Actually the standard tensor decomposition for matrix multiplication:
        # T is a tensor in R^{n x m x p} with T[i,j,k] = 1 for all i,j,k
        # We seek rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i,r] * V[j,r] * W[k,r]
        # But that would give rank 1 if T is all ones, which is wrong.
        #
        # Let me reconsider. The task says:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # And the convention is W[k*n+i, r]
        #
        # So the tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 for all i,j,k (this is the multiplication tensor)
        #
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # Wait, that can't be right because the indices don't match up properly.
        # Let me look at the schoolbook solution:
        # U[i*m+j][col] = 1.0, V[j*p+k][col] = 1.0, W[k*n+i][col] = 1.0
        # So for the schoolbook, each term r corresponds to a specific (i,j,k) triple.
        # T[i',j',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
        # = sum over all r where (i',j',k') matches the (i,j,k) for that r
        # = 1 if there's a term for (i',j',k'), which is always true since we have one for each.
        # So T[i',j',k'] = 1 for all i',j',k'. ✓
        #
        # So the tensor is indeed all ones, and we need:
        # T[i,j,k] = 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # This is a rank decomposition of the all-ones tensor with specific index structure.
        #
        # Actually, I think I'm overcomplicating this. Let me just verify numerically.
        
        # Build the target tensor: T[i,j,k] = 1 for all i,j,k
        # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check shapes
        if U.shape != (n*m, rank):
            return False
        if V.shape != (m*p, rank):
            return False
        if W.shape != (p*n, rank):
            return False
        
        # Verify all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def schoolbook_solution():
        """Return the schoolbook rank-105 solution."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    U_best, V_best, W_best = schoolbook_solution()
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Strategy: Use structured random initialization and iterative refinement
    # The tensor is all-ones with specific index structure, so we can exploit symmetry
    
    # For small ranks, try random initializations with structured patterns
    # Key insight: if we can find a decomposition with rank < 105, it's an improvement
    
    # Try rank values from low to high
    # Known bounds: For n x m x p matrix multiplication, rank is at most n*m*p
    # For 3x5x7, we know rank <= 105
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= n+m+p-2 = 3+5+7-2 = 13 (Bini's bound, but not tight)
    
    # Let's try a few rank values with random search
    # Use the fact that the tensor has a lot of structure
    
    np.random.seed(42)
    
    for target_rank in range(35, min(best_rank, 80)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if target_rank < 50 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with some structure
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Add structured components: try to make each "slice" have some consistency
            # For each i, the rows of U indexed by (i, j) for j=0..m-1 should be related
            # Similarly for other matrices
            
            # Try a few iterations of gradient-like updates
            # Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is non-convex, so use simple iterative updates
            
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient w.r.t. U
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute residual
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residual = val - 1.0
                            
                            # Gradient contributions
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += residual * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += residual * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += residual * U[i*m+j, r] * V[j*p+k, r]
                
                # Update with small step size
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(grad_U)) < 1e-10 and np.max(np.abs(grad_V)) < 1e-10 and np.max(np.abs(grad_W)) < 1e-10:
                    break
            
            # Verify the result
            if verify(U, V, W, tol=1e-6):
                # Found a valid solution with this rank
                best_rank = target_rank
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                break  # Try next rank value (lower)
    
    # Return the best solution found
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
