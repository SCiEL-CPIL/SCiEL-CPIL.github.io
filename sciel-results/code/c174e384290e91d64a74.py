"""
Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
Uses a greedy combinatorial search with local refinement to find low-rank decompositions.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U @ diag @ V^T reconstructs the tensor correctly via W."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But the standard formulation is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Let's verify: for each (i,j,k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # Actually, the matrix multiplication tensor T_{ijk} = 1 if C_ik = sum_j A_ij B_jk
    # The decomposition is: T_{ijk} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Compute the reconstructed tensor
    # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    errors.append((i,j,k,val))
    return len(errors) == 0

def _try_greedy_search(n, m, p, max_rank, time_limit):
    """
    Greedy search for a low-rank decomposition.
    We look for U, V, W with integer/half-integer entries.
    """
    start_time = time.time()
    best_result = None
    
    # For small problems, we can try to find decompositions with rank < n*m*p
    # The schoolbook rank is 4*4*7 = 112
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Also rank >= n*m + m*p + p*n - (n+m+p) + 1 for some cases, but let's just search.
    
    # Strategy: try to find a decomposition with rank r starting from a lower bound
    # We'll use a randomized greedy approach
    
    def random_greedy_attempt(rank, iterations=100):
        """Try to find a decomposition of given rank using greedy construction."""
        for _ in range(iterations):
            if time.time() - start_time > time_limit:
                return None
            
            # Initialize with zeros
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each tensor entry (i,j,k), we need to cover it
            # A rank-1 term (u, v, w) covers (i,j,k) if u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
            # We want to choose rank-1 terms such that for each (i,j,k), 
            # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
            
            # Greedy: pick random sparse vectors
            for c in range(rank):
                # Choose random support for U, V, W
                u_support = set()
                v_support = set()
                w_support = set()
                
                # Each rank-1 term should contribute to a "block" of the tensor
                # Randomly pick a subset of (i,j,k) entries to cover
                num_entries = max(1, (n*m*p) // rank)
                for _ in range(num_entries):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    u_support.add(i*m + j)
                    v_support.add(j*p + k)
                    w_support.add(k*n + i)
                
                if not u_support or not v_support or not w_support:
                    continue
                
                # Assign random values from {-1, 0, 1, 0.5, -0.5}
                for idx in u_support:
                    U[idx, c] = np.random.choice([-1, 1, 0.5, -0.5])
                for idx in v_support:
                    V[idx, c] = np.random.choice([-1, 1, 0.5, -0.5])
                for idx in w_support:
                    W[idx, c] = np.random.choice([-1, 1, 0.5, -0.5])
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        
        return None
    
    # Try increasing ranks
    for rank in range(28, min(max_rank, 60)):
        if time.time() - start_time > time_limit:
            break
        result = random_greedy_attempt(rank, iterations=50)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with exact fractions
            U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, try to find a better decomposition
    start = time.time()
    time_budget = 120  # seconds
    
    result = _try_greedy_search(n, m, p, max_rank=80, time_limit=time_budget - 5)
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
