"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook rank is n*m*p = 4*4*9 = 144.
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
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Condition: sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta(i,i') delta(j,j') delta(k,k')
        """
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Compute T[i, j, j', k, k', i'] = sum_r U[i,j,r] V[j',k,r] W[k',i',r]
        T = np.einsum('ijr,skr,tir->ijstkr', U_3d, V_3d, W_3d, optimize=True)
        
        # Check diagonal: T[i, j, j, k, k, i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, j, k, k, i] - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal: T[i, j, s, k, t, i'] should be 0 if (i,j,k) != (i',s,t)
        ii, jj, ss, kk, tt, ii_prime = np.indices((n, m, m, p, p, n))
        diag_mask = (ii == ii_prime) & (jj == ss) & (kk == tt)
        off_diag = T[~diag_mask]
        if np.any(np.abs(off_diag) > 1e-9):
            return False
        return True

    def verify_fast(U, V, W):
        """
        Faster verification using a different approach:
        For each r, we have a rank-1 tensor contribution.
        The full tensor is sum_r U[:,r] outer V[:,r] outer W[:,r]
        We want this to equal the identity tensor.
        
        Alternative: compute M = U @ V.T @ W.T and check if it matches the expected tensor.
        Actually, let's use the fact that:
        T[i,j,s,k,t,i'] = sum_r U[i*m+j, r] * V[s*p+k, r] * W[t*n+i', r]
        
        We can check this more efficiently by iterating over the 144 required ones
        and a sample of the zeros.
        """
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Check all 144 diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i, j, j, k, k, i] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    val = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check a random sample of off-diagonal entries
        # Total off-diagonal entries: 20736 - 144 = 20592
        # Sample 5000 of them
        sample_size = min(5000, 20592)
        for _ in range(sample_size):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            s = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            t = random.randint(0, p - 1)
            i_prime = random.randint(0, n - 1)
            
            # Skip if it's a diagonal entry
            if i == i_prime and j == s and k == t:
                continue
                
            val = np.sum(U_3d[i, j, :] * V_3d[s, k, :] * W_3d[t, i_prime, :])
            if abs(val) > 1e-9:
                return False
        
        return True

    # Try to find a lower rank decomposition
    # Start from 143 down to 100
    # Use a more structured initialization based on block structure
    
    for target_rank in range(rank - 1, 100, -1):
        if time.time() > deadline:
            break
            
        # Use a smarter initialization: start from schoolbook and perturb
        # Try several different strategies
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            # Strategy: initialize with a subset of schoolbook terms
            # plus some random perturbations
            # Each rank-1 term corresponds to (i,j,k) -> (i',j',k')
            
            # Create a list of possible (i,j,k) indices
            indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(indices)
            
            # Assign each rank to a random (i,j,k) with some probability
            # This creates a sparse structure
            
            for r in range(target_rank):
                # With probability 0.7, use a "clean" schoolbook-like term
                # With probability 0.3, use a random term
                if random.random() < 0.7:
                    i, j, k = indices[r % len(indices)]
                    U_trial[i * m + j, r] = 1
                    V_trial[j * p + k, r] = 1
                    W_trial[k * n + i, r] = 1
                else:
                    # Random sparse term
                    for _ in range(3):  # 3 non-zero entries per matrix
                        if random.random() < 0.8:
                            row = random.randint(0, n * m - 1)
                            U_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
                        if random.random() < 0.8:
                            row = random.randint(0, m * p - 1)
                            V_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
                        if random.random() < 0.8:
                            row = random.randint(0, p * n - 1)
                            W_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
            
            if verify_fast(U_trial, V_trial, W_trial):
                # Confirm with full verification
                if verify(U_trial, V_trial, W_trial):
                    best_rank = target_rank
                    best_result = {
                        "U": U_trial.tolist(),
                        "V": V_trial.tolist(),
                        "W": W_trial.tolist()
                    }
                    break
    
    return best_result
# EVOLVE-BLOCK-END
