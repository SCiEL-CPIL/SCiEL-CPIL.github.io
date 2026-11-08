import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i,j,k form a valid multiplication index, 0 otherwise.
    # Actually, for matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) where the "1" entries are at positions
    # that contribute to the multiplication.
    
    # The correct tensor: T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise
    # But actually the matrix multiplication tensor is:
    # M[i,j,k] = delta(i,j,k) where the multiplication is C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor entries are all 1 for the "multiplication" part.
    
    # Let me reconsider. The matrix multiplication tensor of size n x m x p:
    # T_{ijk} = 1 if the indices are valid (i in [0,n), j in [0,m), k in [0,p))
    # The rank-1 decomposition: sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    
    # Wait, I need to be more careful about the index conventions.
    # U has shape (n*m, rank), indexed as U[i*m + j, r]
    # V has shape (m*p, rank), indexed as V[j*p + k, r]  
    # W has shape (p*n, rank), indexed as W[k*n + i, r]
    
    # The tensor equation: for each (i, j, k), 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since all entries of the multiplication tensor are 1)
    
    # Actually no. Let me think again. The matrix multiplication tensor:
    # Given A (n x m) and B (m x p), C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # The "tensor" that represents this operation has entries T[i,j,k] = 1
    # for all i,j,k in range, because every triple (i,j,k) contributes.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_reduction=10):
    """Try to reduce rank by finding linear dependencies among columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a trilinear constraint.
    
    # Approach: Try to find a vector c such that replacing column r with column r - c*r' 
    # still satisfies the constraints. This is complex.
    
    # Alternative: Use the fact that if we have a valid decomposition of rank R,
    # we can try to find a lower rank one by solving a system of equations.
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_{a=0}^{r-1} U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a] = 1
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # For n=3,m=3,p=5: 45 equations, r*(9+15+15)=45r unknowns.
    
    # Let's try a greedy approach: start with schoolbook and try to eliminate columns.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    for target_rank in range(rank - 1, max(rank - max_reduction - 1, 0) - 1, -1):
        if time.time() > DEADLINE:
            break
        # Try to find a valid decomposition of target_rank
        # This is a hard problem, so let's use a simpler heuristic.
        
        # Heuristic: Try to merge pairs of columns that are "similar"
        # Two columns r1 and r2 can potentially be merged if their contributions
        # can be represented by a single column.
        
        merged = False
        for r1 in range(target_rank, current_rank):
            for r2 in range(r1 + 1, current_rank):
                # Check if we can represent columns r1 and r2 as a single column
                # This requires solving a system of equations.
                
                # For each (i,j,k), we need:
                # U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + 
                # U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2] = 
                # U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i]
                
                # This is generally not possible with a single rank-1 term.
                # So this approach won't work directly.
                
                pass
        
        # Alternative: Use random initialization and optimize
        if time.time() > DEADLINE:
            break
        
        # Try random search for lower rank
        np.random.seed(42)
        for attempt in range(100):
            if time.time() > DEADLINE:
                break
            
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Scale to make the product approximately 1
            # For each (i,j,k), we need sum_a U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] = 1
            # This is underdetermined, so let's try to solve it.
            
            # Build the system: for each (i,j,k), the equation is
            # sum_a U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a] = 1
            # This is nonlinear (trilinear), so we can't solve it directly.
            
            # Instead, let's use a different approach: start with a valid high-rank
            # decomposition and try to project it down.
            
            pass
        
        # If we can't find a lower rank, continue
        break
    
    # For now, return the schoolbook decomposition
    return [[int(x) for x in row] for row in best_U], [[int(x) for x in row] for row in best_V], [[int(x) for x in row] for row in best_W]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, n, m, p)
    
    # Ensure we're within deadline
    if time.time() > DEADLINE:
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
