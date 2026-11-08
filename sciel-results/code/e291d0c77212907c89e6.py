"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    The multiplication tensor T[i,j,k] for (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    We use a systematic search over small ranks with rational/integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with the schoolbook rank = n*m*p = 24
    # Try to find lower ranks
    
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the trivial rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try ranks from low to high
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Also rank >= n*m*p / (some bound) but let's just try
    min_rank = max(n*m, m*p, p*n)  # 12
    
    # Try to find decompositions with rank < n*m*p
    # Use a randomized + structured approach
    
    best_result = None
    best_rank = n * m * p  # start with schoolbook
    
    # First, try some known small-rank constructions
    # For 2x3 * 3x4, the tensor has 24 entries all equal to 1
    
    # Try rank 12, 13, ..., up to 23
    for rank in range(min_rank, n*m*p):
        if time.time() > deadline - 5:
            break
        
        # Use a structured approach: try to find U, V, W with small integer entries
        # such that the tensor equation is satisfied
        
        # Strategy: Fix U and V, solve for W, then check consistency
        # Or use a more direct search
        
        # Try random integer matrices with small entries
        np.random.seed(rank)  # deterministic per rank
        found = False
        
        for attempt in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Generate random small integer matrices
            max_val = 2
            U = np.random.randint(-max_val, max_val+1, (n*m, rank))
            V = np.random.randint(-max_val, max_val+1, (m*p, rank))
            
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of linear equations in W entries
            
            # Build the linear system: for each (i,j,k), 
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            
            num_eqs = n * m * p
            num_unk = p * n * rank  # W has p*n rows and rank columns
            
            # Actually, W is (p*n, rank), so there are p*n*rank unknowns
            # But we have only n*m*p = 24 equations
            # This is underdetermined, so we need additional constraints
            
            # Better approach: try to construct W directly
            # For each column c of W, it's a vector of length p*n
            # The constraint is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # Let's try a different approach: use the fact that we can factor
            # the tensor as a sum of rank-1 tensors
            
            # A rank-1 tensor is T[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            # where a is n*m, b is m*p, c is p*n
            
            # We want to write the all-ones tensor as sum of r rank-1 tensors
            
            # Try greedy: pick random rank-1 components and subtract
            T = np.ones((n, m, p))
            T_remaining = T.copy()
            U_cols = []
            V_cols = []
            W_cols = []
            
            for c in range(rank):
                # Pick a random rank-1 tensor that approximates T_remaining
                # a[i,j] = random, b[j,k] = random, c[k,i] = random
                a = np.random.choice([-1, 1], size=(n, m))
                b = np.random.choice([-1, 1], size=(m, p))
                cc = np.random.choice([-1, 1], size=(p, n))
                
                # Compute the rank-1 tensor
                rank1 = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rank1[i,j,k] = a[i,j] * b[j,k] * cc[k,i]
                
                # Check if adding this helps
                # We want sum of rank1_c to equal T
                # Just accumulate
                U_cols.append(a.flatten().tolist())
                V_cols.append(b.flatten().tolist())
                W_cols.append(cc.flatten().tolist())
            
            # Build U, V, W
            U_mat = np.array(U_cols).T  # (n*m, rank)
            V_mat = np.array(V_cols).T  # (m*p, rank)
            W_mat = np.array(W_cols).T  # (p*n, rank)
            
            # Verify
            if verify_decomposition(U_mat, V_mat, W_mat):
                best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_result is not None:
        return best_result
    else:
        return schoolbook_decomposition()
# EVOLVE-BLOCK-END
