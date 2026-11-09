import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _verify(n, m, p, U, V, W):
    """Verify exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape for efficient computation
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # For each r, the contribution is outer product structure
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(n, m, p, U, V, W):
    """Faster verification using matrix operations."""
    R = len(U[0])
    Ua = np.array(U, dtype=float).reshape(n, m, R)
    Va = np.array(V, dtype=float).reshape(m, p, R)
    Wa = np.array(W, dtype=float).reshape(p, n, R)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For each i,k, check all j
    for i in range(n):
        for k in range(p):
            # For fixed i,k: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # = sum_r W[k,i,r] * U[i,j,r] * V[j,k,r]
            # This is like: for each j, dot product of (U[i,j,:] * V[j,k,:]) with W[k,i,:]
            for j in range(m):
                val = np.dot(Ua[i, j, :] * Va[j, k, :], Wa[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Try structured decompositions first, then local search
    # Key insight: M_{2,4,7} can potentially be decomposed using Strassen-like blocks
    
    # Phase 1: Try Strassen-based block decomposition
    # Split 4x7 into blocks and use 2x2x2 Strassen where possible
    # 4 = 2+2, 7 = 2+2+2+1 (messy) or 7 = 3+4
    
    # Alternative: Use the fact that rank(M_{2,4,7}) <= rank(M_{2,4,4}) + rank(M_{2,4,3})
    # Or use tensor rank bounds
    
    # Phase 2: Local search with better initialization
    # Use random {0, 1, -1} entries and try to minimize error
    
    # For each candidate rank, try multiple random initializations and local search
    # The objective is to make sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Better approach: Use the structure of the problem
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k)
    # This is actually the "all-ones" tensor of shape (n*m) x (m*p) x (p*n) with constraints
    
    # Let's use a gradient-free local search with {0, 1, -1} alphabet
    # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    # For efficiency, compute error incrementally
    
    def compute_error(U, V, W, R):
        """Compute total squared error."""
        err = 0.0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j, :] * Va[j*p+k, :], Wa[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    # Try ranks from 32 to 55
    # Use {0, 1, -1} alphabet for entries
    for R in range(32, rank_sb + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Multiple trials for each rank
        max_trials = 5 if R <= 40 else 3
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with {0, 1, -1}
            U = np.random.choice([-1, 0, 1], size=(n * m, R)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m * p, R)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p * n, R)).astype(float)
            
            # Ensure some non-zero entries
            # Local search: perturb random entries
            best_err = compute_error(U, V, W, R)
            
            if best_err < 1e-9:
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                return {"U": best_U, "V": best_V, "W": best_W}
            
            # Local search
            for step in range(200):
                if time.time() > deadline - 3.0:
                    break
                
                # Pick a random matrix and entry to perturb
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    r_idx = np.random.randint(0, n * m)
                    c_idx = np.random.randint(0, R)
                    old_val = U[r_idx, c_idx]
                    # Try adding or removing
                    new_val = old_val + np.random.choice([-1, 1])
                    if new_val not in [-1, 0, 1]:
                        new_val = -old_val if old_val != 0 else 1
                    U[r_idx, c_idx] = new_val
                elif mat_choice == 1:
                    r_idx = np.random.randint(0, m * p)
                    c_idx = np.random.randint(0, R)
                    old_val = V[r_idx, c_idx]
                    new_val = old_val + np.random.choice([-1, 1])
                    if new_val not in [-1, 0, 1]:
                        new_val = -old_val if old_val != 0 else 1
                    V[r_idx, c_idx] = new_val
                else:
                    r_idx = np.random.randint(0, p * n)
                    c_idx = np.random.randint(0, R)
                    old_val = W[r_idx, c_idx]
                    new_val = old_val + np.random.choice([-1, 1])
                    if new_val not in [-1, 0, 1]:
                        new_val = -old_val if old_val != 0 else 1
                    W[r_idx, c_idx] = new_val
                
                new_err = compute_error(U, V, W, R)
                
                if new_err < 1e-9:
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}
                
                if new_err < best_err:
                    best_err = new_err
                else:
                    # Revert
                    if mat_choice == 0:
                        U[r_idx, c_idx] = old_val
                    elif mat_choice == 1:
                        V[r_idx, c_idx] = old_val
                    else:
                        W[r_idx, c_idx] = old_val
    
    # Phase 3: Try a more structured approach
    # Use the schoolbook but try to merge terms
    # Actually, let's try a different initialization strategy:
    # Start from schoolbook and try to zero out entries while maintaining correctness
    
    if time.time() < deadline - 10.0:
        # Try to find a lower rank by starting from schoolbook and removing columns
        # This is a heuristic: try random subsets of columns and check if they span
        # Actually, this doesn't directly work for tensor rank
        
        # Alternative: Use the known result that rank(M_{2,4,7}) <= 35
        # Try to construct a rank-35 decomposition using algebraic methods
        # For now, rely on the local search above
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
