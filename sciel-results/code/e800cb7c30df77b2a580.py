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
    deadline = time.time() + 140  # Safety margin before 150s limit
    
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
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Compute T[i, j, j', k, k', i'] = sum_r U[i,j,r] V[j',k,r] W[k',i',r]
        # Note: V index is (j', k), W index is (k', i')
        # np.einsum('ijr,skr,tir->ijstkr', U, V, W) ?
        # Let's map indices:
        # U: i, j, r
        # V: s, k, r  (s corresponds to j')
        # W: t, i, r  (t corresponds to k', second index is i')
        # We want result indexed by i, j, s, k, t, i'
        # So: 'ijr,skr,tir->ijstkr'
        
        T = np.einsum('ijr,skr,tir->ijstkr', U_3d, V_3d, W_3d, optimize=True)
        
        # Check diagonal: T[i, j, j, k, k, i] should be 1
        # Indices: i, j, s=j, k, t=k, i'=i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, j, k, k, i] - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal: T[i, j, s, k, t, i'] should be 0 if (i,j,k) != (i',s,t)
        # We can check all entries
        # Create a mask of valid diagonal entries
        # Or just check all and ensure they are 0 unless on diagonal
        # Since T is small (20736 entries), we can iterate or use vectorized check
        
        # Vectorized check:
        # Create indices
        ii, jj, ss, kk, tt, ii_prime = np.indices((n, m, m, p, p, n))
        
        # Diagonal mask: i==i', j==s, k==t
        diag_mask = (ii == ii_prime) & (jj == ss) & (kk == tt)
        
        # Off-diagonal entries must be 0
        off_diag = T[~diag_mask]
        if np.any(np.abs(off_diag) > 1e-9):
            return False
            
        return True

    # Try to find a lower rank decomposition
    # Start from 143 down to a reasonable limit
    for target_rank in range(rank - 1, 120, -1):
        if time.time() > deadline:
            break
            
        # Try a few random attempts
        # We use integer/half-integer entries
        values = [1, -1, 0.5, -0.5]
        
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            # Sparse random initialization
            for r in range(target_rank):
                # U
                if random.random() < 0.8:
                    row = random.randint(0, n * m - 1)
                    U_trial[row, r] = random.choice(values)
                # V
                if random.random() < 0.8:
                    row = random.randint(0, m * p - 1)
                    V_trial[row, r] = random.choice(values)
                # W
                if random.random() < 0.8:
                    row = random.randint(0, p * n - 1)
                    W_trial[row, r] = random.choice(values)
            
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
