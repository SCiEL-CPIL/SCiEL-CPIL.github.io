import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
DEADLINE = 140
SAFETY_MARGIN = 10

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36
    
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = rank_schoolbook
    
    def verify_full(U, V, W, rank):
        """Full verification: check all (i,j,k) vs (i',j',k') combinations."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        # Check half-integer constraint
        for row in U:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        for row in V:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        for row in W:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        
        # Check all tensor entries
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                s = 0.0
                                for c in range(rank):
                                    s += U_arr[i * m + j, c] * V_arr[jp * p + k, c] * W_arr[kp * n + ip, c]
                                if abs(s - expected) > 1e-8:
                                    return False
        return True
    
    def try_round_and_verify(U_float, V_float, W_float, rank):
        """Try rounding to half-integers and verify."""
        U_r = [[round(x * 2) / 2 for x in row] for row in U_float]
        V_r = [[round(x * 2) / 2 for x in row] for row in V_float]
        W_r = [[round(x * 2) / 2 for x in row] for row in W_float]
        
        if verify_full(U_r, V_r, W_r, rank):
            return U_r, V_r, W_r
        return None
    
    # Strategy: Use structured search based on algebraic properties
    # For 2x3x6, try to find rank < 36 decompositions
    
    rng = np.random.RandomState(42)
    
    # Try various target ranks from low to high
    for target_rank in range(24, 37):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        max_trials = 50 if target_rank <= 30 else 20
        
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Initialize with structured random values
            U_init = rng.randn(n * m, target_rank) * 0.3
            V_init = rng.randn(m * p, target_rank) * 0.3
            W_init = rng.randn(p * n, target_rank) * 0.3
            
            # Normalize each column to have unit geometric mean
            for c in range(target_rank):
                u_norm = np.linalg.norm(U_init[:, c])
                v_norm = np.linalg.norm(V_init[:, c])
                w_norm = np.linalg.norm(W_init[:, c])
                if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                    scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                    U_init[:, c] /= scale
                    V_init[:, c] /= scale
                    W_init[:, c] /= scale
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            max_iter = 500
            prev_loss = float('inf')
            stagnation_count = 0
            
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                # Compute loss and gradients efficiently
                gU = np.zeros_like(U_curr)
                gV = np.zeros_like(V_curr)
                gW = np.zeros_like(W_curr)
                loss = 0.0
                
                # Compute all tensor entries
                for i in range(n):
                    for ip in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    for kp in range(p):
                                        expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        total = 0.0
                                        for c in range(target_rank):
                                            total += U_curr[i * m + j, c] * V_curr[jp * p + k, c] * W_curr[kp * n + ip, c]
                                        error = total - expected
                                        loss += error * error
                                        if abs(error) > 1e-10:
                                            for c in range(target_rank):
                                                gU[i * m + j, c] += 2 * error * V_curr[jp * p + k, c] * W_curr[kp * n + ip, c]
                                                gV[jp * p + k, c] += 2 * error * U_curr[i * m + j, c] * W_curr[kp * n + ip, c]
                                                gW[kp * n + ip, c] += 2 * error * U_curr[i * m + j, c] * V_curr[jp * p + k, c]
                
                # Update
                U_curr -= lr * gU
                V_curr -= lr * gV
                W_curr -= lr * gW
                
                # Adaptive learning rate
                if loss < prev_loss * 0.999:
                    lr = min(lr * 1.05, 0.5)
                    stagnation_count = 0
                else:
                    lr *= 0.95
                    stagnation_count += 1
                
                prev_loss = loss
                
                if loss < 1e-10:
                    break
                if stagnation_count > 100:
                    break
            
            # Try rounding
            result = try_round_and_verify(U_curr, V_curr, W_curr, target_rank)
            if result is not None:
                U_r, V_r, W_r = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    if target_rank <= 28:
                        break
        
        if best_rank <= 28:
            break
    
    return best_result
