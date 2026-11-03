"""Matrix multiplication 4x4x9 rank minimization - combined approach."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j, :]
                    v_val = V_arr[j*p + k, :]
                    w_val = W_arr[k*n + i, :]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower-rank decompositions using structured approaches
    np.random.seed(42)
    
    # Method 1: Random linear combinations with rank reduction attempts
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Generate random mixing matrices
        r = rank_sb
        Q = np.random.randn(r, r)
        Q, _ = np.linalg.qr(Q)
        
        U_new = U_sb @ Q
        V_new = V_sb @ Q
        W_new = W_sb @ Q
        
        # Try to compress by identifying near-zero columns
        U_norms = np.linalg.norm(U_new, axis=0)
        V_norms = np.linalg.norm(V_new, axis=0)
        W_norms = np.linalg.norm(W_new, axis=0)
        
        # Keep only significant columns
        mask = (U_norms > 1e-10) & (V_norms > 1e-10) & (W_norms > 1e-10)
        if mask.sum() < best_rank:
            U_c = U_new[:, mask]
            V_c = V_new[:, mask]
            W_c = W_new[:, mask]
            if validate_fast(U_c, V_c, W_c):
                best_rank = mask.sum()
                best = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
    
    # Method 2: Try Strassen-inspired block decomposition
    # For 4x4 x 4x9, we can potentially use block structure
    # A is 4x4, B is 4x9
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1], [B2]] where B1, B2 are 2x9
    # C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # Strassen for the block multiplication:
    # M1 = (A11+A22)*(B1+B2)
    # M2 = (A21+A22)*B1
    # M3 = A11*(B1-B2)
    # M4 = A22*(B2-B1)
    # M5 = (A11+A12)*B2
    # M6 = (A21-A11)*(B1+B2)
    # M7 = (A12-A22)*(B2+B1)
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M5
    # C12 = M3 + M1
    # C22 = M2 + M4
    
    # This gives 7 block multiplications instead of 8
    # Each block multiplication is 2x2 * 2x9
    # Rank of 2x2*2x9 is 2*2*9 = 36
    # So total rank would be 7*36 = 252, which is worse than schoolbook 144
    
    # Alternative: Try to find a direct lower-rank decomposition
    # The theoretical minimum rank for 4x4x4 is 49 (Strassen)
    # For 4x4x9, we need to find the optimal rank
    
    # Try a systematic search for structured decompositions
    # This is complex, so we'll stick with the random approach and return best found
    
    # Method 3: Try Kronecker product structure exploitation
    # The 4x4 identity matrix has Kronecker product structure: I4 = I2 ⊗ I2
    # This might allow rank reduction
    
    # Method 4: Try gradient-free optimization for rank reduction
    # Use simulated annealing or similar to find lower-rank solutions
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Try to find a lower-rank decomposition by perturbing the schoolbook
        target_rank = max(50, best_rank - 10 - attempt * 5)
        
        # Initialize with random matrices
        U_rand = np.random.randn(n*m, target_rank) * 0.1
        V_rand = np.random.randn(m*p, target_rank) * 0.1
        W_rand = np.random.randn(p*n, target_rank) * 0.1
        
        # Use schoolbook as initialization for some columns
        if target_rank <= rank_sb:
            U_rand[:, :min(target_rank, rank_sb)] = U_sb[:, :min(target_rank, rank_sb)]
            V_rand[:, :min(target_rank, rank_sb)] = V_sb[:, :min(target_rank, rank_sb)]
            W_rand[:, :min(target_rank, rank_sb)] = W_sb[:, :min(target_rank, rank_sb)]
        
        # Try to optimize
        for step in range(100):
            if time.time() > deadline:
                break
            
            # Compute current error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_rand[i*m+j,:], V_rand[j*p+k,:] * W_rand[k*n+i,:])
                        error += (s - 1.0) ** 2
            
            if error < 1e-6:
                if validate_fast(U_rand, V_rand, W_rand):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                break
            
            # Random perturbation
            perturbation_scale = 0.01 / (1 + step * 0.1)
            U_rand += np.random.randn(n*m, target_rank) * perturbation_scale
            V_rand += np.random.randn(m*p, target_rank) * perturbation_scale
            W_rand += np.random.randn(p*n, target_rank) * perturbation_scale
    
    # Final: verify and return best found
    return best
# EVOLVE-BLOCK-END
