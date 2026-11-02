"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p  # 72
    
    # Schoolbook decomposition as baseline
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Strategy: Use the Khatri-Rao approach
    # Fix U and V, solve for W via linear system
    # K * W^T = T_flat where K = [U[:,c] ⊗ V[:,c] for c in range(r)]
    
    T_flat = T.reshape(nm * mp, pn)  # (216, 24)
    
    # Try to find lower rank using structured random search
    # Key insight: allow U, V to have entries that create cancellation
    
    # For each candidate rank, try multiple random initializations
    # Use a mix of small integers and half-integers
    
    for target_rank in range(71, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts scales with how aggressive the rank reduction is
        max_attempts = min(20, max(3, 100 // target_rank))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            
            # Random initialization with small values
            # Use a mixture: some entries are 0, ±1, ±0.5
            U_rand = np.zeros((nm, target_rank))
            V_rand = np.zeros((mp, target_rank))
            
            for c in range(target_rank):
                # Each column of U has sparse structure
                # Randomly choose which rows are nonzero
                u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
                v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=mp)
                # Zero out some entries to create structure
                u_mask = np.random.random(nm) < 0.5
                v_mask = np.random.random(mp) < 0.5
                U_rand[:, c] = np.where(u_mask, u_vals, 0.0)
                V_rand[:, c] = np.where(v_mask, v_vals, 0.0)
            
            # Form Khatri-Rao matrix
            K = np.zeros((nm * mp, target_rank))
            for c in range(target_rank):
                K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
            
            # Solve K * W^T = T_flat
            try:
                W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
            except np.linalg.LinAlgError:
                continue
            
            # Check exactness
            residual = np.abs(K @ W_sol - T_flat).max()
            
            if residual < 1e-9:
                # Check if entries are close to half-integers
                W_check = W_sol * 2
                if np.allclose(W_check, np.round(W_check), atol=1e-6):
                    W_final = W_sol.T
                    # Round to nearest half-integer
                    W_final = np.round(W_final * 2) / 2
                    U_final = U_rand
                    V_final = V_rand
                    
                    # Final verification
                    recon = np.zeros((nm, mp, pn))
                    for c in range(target_rank):
                        recon += np.einsum('a,b,c->abc', U_final[:,c], V_final[:,c], W_final[:,c])
                    if np.abs(recon - T).max() < 1e-8:
                        best_rank = target_rank
                        best_decomp = (U_final, V_final, W_final)
                        break
        
        if best_rank < schoolbook_rank:
            break
    
    # Also try a direct structured approach:
    # For the 3x3x8 case, try to exploit the fact that p=8 is composite
    # Split into two 3x3x4 problems if possible
    if best_rank == schoolbook_rank and time.time() < deadline - 20:
        # Try rank 36 (two independent 3x3x4 schoolbook)
        # This won't help, but try rank 30, 27, etc. with better search
        for target_rank in [36, 30, 27, 24]:
            if time.time() > deadline - 10:
                break
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                np.random.seed(999 + target_rank * 100 + attempt)
                U_rand = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(nm, target_rank))
                V_rand = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(mp, target_rank))
                K = np.zeros((nm * mp, target_rank))
                for c in range(target_rank):
                    K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
                try:
                    W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
                except:
                    continue
                residual = np.abs(K @ W_sol - T_flat).max()
                if residual < 1e-9:
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        W_final = np.round(W_sol.T * 2) / 2
                        recon = np.zeros((nm, mp, pn))
                        for c in range(target_rank):
                            recon += np.einsum('a,b,c->abc', U_rand[:,c], V_rand[:,c], W_final[:,c])
                        if np.abs(recon - T).max() < 1e-8:
                            best_rank = target_rank
                            best_decomp = (U_rand.copy(), V_rand.copy(), W_final)
                            break
            if best_rank < schoolbook_rank:
                break
    
    U_out, V_out, W_out = best_decomp
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
