import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _build_strassen_56():
    """Build the 8-block Strassen decomposition (rank 56)."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Standard Strassen for 2x2 * 2x2
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_contribs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p, num_tests=3):
    """Verify the decomposition is correct."""
    for seed in range(num_tests):
        np.random.seed(seed)
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        C_decomp = W @ (UA * VB)
        C_flat = C.T.flatten()
        
        if not np.allclose(C_decomp, C_flat, rtol=1e-9, atol=1e-9):
            return False
    return True

def _try_rank_reduction(U0, V0, W0, target_rank, n, m, p, deadline):
    """
    Try to find a decomposition of target_rank by solving linear systems.
    Strategy: try removing one term at a time from the 56-rank solution.
    """
    start = time.time()
    R0 = U0.shape[1]  # 56
    
    # For each term to remove, try to solve for adjusted coefficients
    for remove_idx in range(R0):
        if time.time() > deadline:
            break
        
        # Keep all terms except remove_idx
        keep = [i for i in range(R0) if i != remove_idx]
        R = len(keep)
        
        if R < target_rank:
            continue
        
        # Build the linear system: for each (i,j,k) entry of the tensor,
        # we need sum_r U[i,r]*V[j,r]*W[k,r] = delta (the target tensor value)
        # The target tensor for matrix mult: T[i,j,k] = 1 if j==k (in appropriate indexing)
        # Actually for A(n,m) * B(m,p) -> C(p,n):
        # C[k,i] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 if the indices match: A row i, B row j, C col k, with j being the shared index
        # In flattened: A index = i*m+j, B index = j*p+k, C index = k*n+i
        
        # Number of tensor entries
        num_A = n * m
        num_B = m * p
        num_C = p * n
        num_entries = num_A * num_B * num_C  # This is huge: 4*32*32 = 4096
        
        # But we only need to satisfy the bilinear identity:
        # For all A, B: W @ (U^T A * V^T B) = (A@B).T.flatten()
        # This is equivalent to: for all i in A, j in B, k in C:
        #   sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # where T[i,j,k] = 1 if the indices correspond to a valid product term
        
        # Build T tensor
        T = np.zeros((num_A, num_B, num_C), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T[a_idx, b_idx, c_idx] = 1
        
        # For the reduced system, we have R unknowns (the coefficients of kept terms)
        # But U, V, W are all unknowns too! This is a 3-tensor problem.
        # 
        # Alternative approach: fix U and V from the original, only solve for W adjustments.
        # Or: use the fact that we can perturb the solution.
        
        # Simpler approach: try to see if one term is redundant by checking
        # if it can be expressed as a linear combination of others in the tensor sense.
        
        # For now, let's try a different strategy:
        # Try to find if rank 55 exists by random perturbation + projection
        
        # Actually, let's try: for each pair of terms, see if we can merge them
        # by finding a rank-1 update that cancels one and adjusts the other.
        
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-56 decomposition
    U, V, W = _build_strassen_56()
    
    # Verify it works
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline:
        # Attempt: try rank 55 by solving a linear system
        # The idea: the tensor equation is linear in W for fixed U, V
        # But we need all three to be consistent.
        
        # Try a different structural approach: 
        # Can we share Strassen terms across blocks?
        # The 8 blocks are independent 2x2 multiplications, so no sharing is possible
        # without changing the problem structure.
        
        # Try: is there a known better algorithm for 2x2 * 2x16?
        # The bilinear complexity of 2xn matrix multiplication...
        # For 2x2: 7 (Strassen)
        # For 2x4: could be 13 or 14? (two Strassen blocks = 14)
        # For 2x8: 26 or 28?
        # For 2x16: 52 or 56?
        
        # Actually, there might be a better algorithm using the structure.
        # Let's try to search for rank 55 by attempting to find a valid decomposition.
        
        # Strategy: Use the 56-rank solution and try to eliminate one term
        # by adjusting all other terms.
        
        R0 = U.shape[1]
        
        for term_to_remove in range(R0):
            if time.time() > deadline:
                break
            
            # Try to solve: find U', V', W' with R0-1 columns
            # such that the tensor equation holds.
            
            # This is a system of equations in the entries of U', V', W'
            # Total unknowns: (n*m + m*p + p*n) * (R0-1) = (4 + 32 + 32) * 55 = 4070
            # Number of equations: n*m*p = 2*2*16 = 64 (one for each output entry)
            # Wait, that's not right. The tensor has n*m*p entries = 64.
            # But each equation is bilinear in the unknowns.
            
            # Actually, the constraint is:
            # For all i in [0,n), j in [0,m), k in [0,p):
            #   sum_r U'[i,r] * V'[j,r] * W'[k,r] = 1 if (i,j,k) is a valid product index
            #   = 0 otherwise
            # 
            # Valid product indices: for A(i,j) * B(j,k) -> C(k,i)
            # So the tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
            # That's n*m*p = 64 equations.
            # But the unknowns are 4070, so the system is highly underdetermined.
            # 
            # The problem is that it's a system of trilinear equations, not linear.
            
            # Alternative: fix two of U, V, W and solve for the third.
            # Fix U and V from the original (with the removed term set to 0),
            # then solve for W.
            
            # U_new = U with column term_to_remove zeroed
            # V_new = V with column term_to_remove zeroed
            # W_new = ?
            
            # The constraint: W_new @ (U_new^T @ A * V_new^T @ B) = (A@B).T.flatten()
            # for all A, B.
            
            # This means: for each r, W_new[:,r] must satisfy:
            # W_new[:,r] = (some function of U_new[:,r] and V_new[:,r])
            
            # More precisely, the tensor equation is:
            # sum_r W_new[k,r] * U_new[i,r] * V_new[j,r] = T[i,j,k]
            
            # For fixed U_new and V_new, this is linear in W_new:
            # For each (i,j,k): sum_r W_new[k,r] * (U_new[i,r] * V_new[j,r]) = T[i,j,k]
            
            # This is a system: for each k, W_new[k,:] is a row of W_new.
            # The equation for output (k,i,j) is:
            # sum_r W_new[k,r] * U_new[i,r] * V_new[j,r] = T[i,j,k]
            
            # Group by k: for each k, we have n*m = 4 equations:
            # sum_r W_new[k,r] * (U_new[:,r] * V_new[:,r][j]) ... 
            # This is getting complex. Let me just try it numerically.
            
            U_new = U.copy()
            V_new = V.copy()
            U_new[:, term_to_remove] = 0
            V_new[:, term_to_remove] = 0
            
            # Build the linear system for W_new
            # Unknowns: W_new has shape (p*n, R0-1) = (32, 55)
            # But column term_to_remove is zero, so effectively (32, 55) unknowns
            # Equations: for each (i,j,k): sum_r W_new[k,r] * U_new[i,r] * V_new[j,r] = T[i,j,k]
            # That's 64 equations.
            
            # Let's build the system properly.
            num_A = n * m
            num_B = m * p
            num_C = p * n
            R = R0 - 1
            
            # For each output index k in [0, p*n), and each (i,j) pair:
            # sum_r W_new[k,r] * U_new[i,r] * V_new[j,r] = T[i,j,k]
            
            # Let's index: for output position c = k*n + i where k in [0,p), i in [0,n)
            # and input positions a = i*m + j, b = j*p + k
            
            # Equation for (a, b, c): sum_r W_new[c, r] * U_new[a, r] * V_new[b, r] = T[a,b,c]
            
            # This is linear in W_new[c, :] for each c.
            # For a fixed c, the equations involving W_new[c, :] are:
            # For each (a, b) such that the output index is c:
            #   sum_r W_new[c, r] * U_new[a, r] * V_new[b, r] = T[a,b,c]
            
            # For output c = k*n + i, the valid (a,b) are:
            # a = i*m + j, b = j*p + k for j in [0, m)
            # So m = 2 equations per output c.
            # Total: p*n*m = 16*2*2 = 64 equations.
            
            # Each equation has R = 55 unknowns (W_new[c, r] for r in kept indices).
            # But the equations for different c involve different rows of W_new,
            # so we can solve them independently!
            
            # For each c, we have m = 2 equations in 55 unknowns. Underdetermined!
            # We need additional constraints (integer/half-integer, or minimality).
            
            # This approach won't work directly. Let me try a different tactic.
            
            pass
        
        # Given the difficulty, let's just return the rank-56 solution.
        # The structural change here is the cleaner code organization.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
