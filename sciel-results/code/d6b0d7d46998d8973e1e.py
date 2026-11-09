"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_pattern(n, m, p, rank, deadline):
    """
    Try specific structured patterns for a given rank.
    Returns a valid decomposition dict or None.
    """
    if time.time() > deadline:
        return None
    
    # Pattern 1: Block structure - for each j, use a subset of rank terms
    # Try: rank = m * r_per_j where each j-block uses r_per_j terms
    # For the j-block, we need to represent an n x p "identity-like" tensor
    # T[i,j,k] for fixed j is 1 for all i,k
    
    # Actually, let's try: use rank terms where each term (c) has support
    # U[i*m+j, c] nonzero only for certain (i,j)
    # V[j*p+k, c] nonzero only for certain (j,k)  
    # W[k*n+i, c] nonzero only for certain (k,i)
    
    # Pattern: sparse support patterns
    # Each column c defines a "factor" that picks out a set of (i,j), (j,k), (k,i)
    
    # Try: each factor picks out one i, one j, one k (schoolbook-like but with sharing)
    # Or: each factor picks out a row of i for a given j, and a column of k for a given j
    
    # Let's try a systematic sparse pattern search
    # For small ranks, enumerate possible support patterns
    
    if rank <= 40:
        # For small ranks, try to find a valid assignment
        # Use a more structured approach:
        
        # Idea: group by j. For each j, we need to cover all (i,k) pairs.
        # If we use r_j terms for index j, those terms must have
        # U[i*m+j, c] and V[j*p+k, c] and W[k*n+i, c] nonzero appropriately.
        
        # For a term c associated with index j:
        # U[i*m+j, c] = a_{i,c} for i in 0..n-1
        # V[j*p+k, c] = b_{k,c} for k in 0..p-1
        # W[k*n+i, c] = w_{k,i,c}
        # We need: sum_{c in j-group} a_{i,c} * b_{k,c} * w_{k,i,c} = 1 for all i,k
        
        # This is still hard. Let's try: w_{k,i,c} = 1/(a_{i,c}*b_{k,c}) if both nonzero
        # Then sum_c 1 = number of terms = 1? No, that gives r_j.
        
        # Alternative: use a_{i,c} = delta_{i, i_c}, b_{k,c} = delta_{k, k_c}
        # Then w_{k,i,c} = 1 if k=k_c and i=i_c, else 0
        # Sum over c in j-group: only term where i_c=i and k_c=k contributes = 1
        # But this requires r_j >= n*p for each j, total rank >= m*n*p = schoolbook
        
        # So we need to share terms across j indices or use non-sparse patterns.
        
        # Let's try: for each term c, allow it to be used by multiple j values.
        # U[i*m+j, c] can be nonzero for multiple j.
        
        # New idea: use rank = n*p + m - 1 or similar
        # For each (i,k), have a term that handles the "diagonal" and use m-1 extra terms
        
        # Actually, let's try a different approach:
        # Use the fact that we can write 1 = sum of products in many ways.
        
        # Try: rank = n*p*m / g where g is some gcd-like factor
        # Or try specific small ranks with random search but better structured
        
        # Let's do a smarter random search:
        # Initialize with sparse random patterns
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            np.random.seed(attempt * 7919 + rank * 104729)
            
            # Create sparse random matrices with half-integer entries
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each column has a few nonzeros
            nnz_per_col = max(1, min(3, rank // 3))
            
            for c in range(rank):
                # Choose which rows of U are nonzero
                u_rows = np.random.choice(n*m, size=min(nnz_per_col, n*m), replace=False)
                for r in u_rows:
                    U[r, c] = np.random.choice([-1, -0.5, 0.5, 1])
                
                v_rows = np.random.choice(m*p, size=min(nnz_per_col, m*p), replace=False)
                for r in v_rows:
                    V[r, c] = np.random.choice([-1, -0.5, 0.5, 1])
                
                w_rows = np.random.choice(p*n, size=min(nnz_per_col, p*n), replace=False)
                for r in w_rows:
                    W[r, c] = np.random.choice([-1, -0.5, 0.5, 1])
            
            U_list = [[U[i,j] for j in range(rank)] for i in range(n*m)]
            V_list = [[V[i,j] for j in range(rank)] for i in range(m*p)]
            W_list = [[W[i,j] for j in range(rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Pattern 2: Use the structure where we factor through a common basis
    # Try: U[i*m+j, c] = u_i * v_j, V[j*p+k, c] = v_j * w_k, W[k*n+i, c] = w_k * u_i / (v_j^2)
    # This requires v_j != 0 and gives sum_c u_i*v_j * v_j*w_k * w_k*u_i/v_j^2 = sum_c u_i^2 * w_k^2
    # = (sum_c u_i^2)(sum_c w_k^2) which is independent of j. Not quite.
    
    # Pattern 3: Direct construction for specific ranks
    # Try rank = m * n * p / 2, m * n * p / 3, etc. with structured merging
    
    # For now, return None if no pattern worked
    return None

def _try_low_rank_search(n, m, p, deadline):
    """
    Systematically search for low-rank decompositions.
    """
    schoolbook_rank = n * m * p
    
    # Known lower bound for matrix mult rank is at least max(n*m, m*p, p*n)
    min_possible = max(n*m, m*p, p*n)
    
    # Try ranks from low to high
    for rank in range(min_possible, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        result = _try_structured_pattern(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin for 150s limit
    
    # First, get the schoolbook solution as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    result = _try_low_rank_search(n, m, p, deadline)
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return fallback
# EVOLVE-BLOCK-END
