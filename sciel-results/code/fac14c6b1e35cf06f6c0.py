import time
import sys
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    block_size = 3
    dim = 6
    
    # Strassen steps for 2x2 blocks
    steps = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}}
    ]
    
    def build_decomposition(inner_U, inner_V, inner_W, inner_rank):
        """Build full 6x6 decomposition from 3x3 inner decomposition.
        inner_U, inner_V, inner_W are 3D arrays of shape (3,3,inner_rank)
        """
        total_rank = 7 * inner_rank
        U = [[0] * total_rank for _ in range(36)]
        V = [[0] * total_rank for _ in range(36)]
        W = [[0] * total_rank for _ in range(36)]
        
        col = 0
        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            
            for inner_r in range(inner_rank):
                # U contribution
                for (I, J), ca in A_coeffs.items():
                    if ca != 0:
                        for u in range(block_size):
                            for v in range(block_size):
                                i_g = I * block_size + u
                                j_g = J * block_size + v
                                idx_u = i_g * dim + j_g
                                val = ca * inner_U[u, v, inner_r]
                                if val != 0:
                                    U[idx_u][col] += val
                
                # V contribution
                for (J, K), cb in B_coeffs.items():
                    if cb != 0:
                        for v in range(block_size):
                            for w in range(block_size):
                                j_g = J * block_size + v
                                k_g = K * block_size + w
                                idx_v = j_g * dim + k_g
                                val = cb * inner_V[v, w, inner_r]
                                if val != 0:
                                    V[idx_v][col] += val
                
                # W contribution
                for (I, K), cc in C_coeffs.items():
                    if cc != 0:
                        for u in range(block_size):
                            for w in range(block_size):
                                i_g = I * block_size + u
                                k_g = K * block_size + w
                                idx_w = k_g * dim + i_g
                                val = cc * inner_W[u, w, inner_r]
                                if val != 0:
                                    W[idx_w][col] += val
                
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct"""
        rank = len(U[0])
        T = np.zeros((6, 6, 6))
        for r in range(rank):
            for i in range(6):
                for j in range(6):
                    u_val = U[i * 6 + j][r]
                    if u_val == 0:
                        continue
                    for k in range(6):
                        v_val = V[j * 6 + k][r]
                        if v_val == 0:
                            continue
                        w_val = W[k * 6 + i][r]
                        if w_val == 0:
                            continue
                        T[i, j, k] += u_val * v_val * w_val
        
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    expected = 1.0 if i == k else 0.0
                    if abs(T[i, j, k] - expected) > 1e-9:
                        return False
        return True
    
    # Schoolbook 3x3: rank 27
    inner_rank_school = 27
    inner_U_sb = np.zeros((3, 3, inner_rank_school))
    inner_V_sb = np.zeros((3, 3, inner_rank_school))
    inner_W_sb = np.zeros((3, 3, inner_rank_school))
    
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                inner_U_sb[i, j, r] = 1.0
                inner_V_sb[j, k, r] = 1.0
                inner_W_sb[i, k, r] = 1.0
                r += 1
    
    # Build and verify schoolbook result
    result_school = build_decomposition(inner_U_sb, inner_V_sb, inner_W_sb, inner_rank_school)
    
    # Verify it's correct
    if not verify(result_school['U'], result_school['V'], result_school['W']):
        # Fallback: pure schoolbook 6x6 (rank 216)
        U = [[0] * 216 for _ in range(36)]
        V = [[0] * 216 for _ in range(36)]
        W = [[0] * 216 for _ in range(36)]
        r = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i * 6 + j][r] = 1.0
                    V[j * 6 + k][r] = 1.0
                    W[k * 6 + i][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better 3x3 decomposition using optimization
    if time.time() < deadline - 60:
        # Try random search for lower rank 3x3 decomposition
        # This is computationally expensive, so try a few random attempts
        import random
        random.seed(42)
        
        best_result = result_school
        best_rank = 27
        
        # Try rank 25
        for target_rank in [25, 24, 23]:
            if time.time() > deadline - 30:
                break
            
            # Try a few random initializations
            for attempt in range(10):
                if time.time() > deadline - 30:
                    break
                
                # Initialize with random values
                inner_U_try = np.random.randn(3, 3, target_rank)
                inner_V_try = np.random.randn(3, 3, target_rank)
                inner_W_try = np.random.randn(3, 3, target_rank)
                
                # Simple gradient-free search: try to minimize error
                # This is a non-convex problem, so use random restarts
                best_err = float('inf')
                best_params = None
                
                for step in range(100):
                    # Random perturbation
                    if step % 10 == 0:
                        inner_U_try = inner_U_try + np.random.randn(3, 3, target_rank) * 0.1
                        inner_V_try = inner_V_try + np.random.randn(3, 3, target_rank) * 0.1
                        inner_W_try = inner_W_try + np.random.randn(3, 3, target_rank) * 0.1
                    
                    # Build and check error
                    U_try, V_try, W_try = build_decomposition(inner_U_try, inner_V_try, inner_W_try, target_rank)['U'], \
                                             build_decomposition(inner_U_try, inner_V_try, inner_W_try, target_rank)['V'], \
                                             build_decomposition(inner_U_try, inner_V_try, inner_W_try, target_rank)['W']
                    
                    # Compute error
                    T = np.zeros((6, 6, 6))
                    for r_idx in range(7 * target_rank):
                        for i in range(6):
                            for j in range(6):
                                u_val = U_try[i * 6 + j][r_idx]
                                if u_val == 0:
                                    continue
                                for k in range(6):
                                    v_val = V_try[j * 6 + k][r_idx]
                                    if v_val == 0:
                                        continue
                                    w_val = W_try[k * 6 + i][r_idx]
                                    if w_val == 0:
                                        continue
                                    T[i, j, k] += u_val * v_val * w_val
                    
                    err = 0.0
                    for i in range(6):
                        for j in range(6):
                            for k in range(6):
                                expected = 1.0 if i == k else 0.0
                                err += (T[i, j, k] - expected) ** 2
                    
                    if err < best_err:
                        best_err = err
                        best_params = (inner_U_try.copy(), inner_V_try.copy(), inner_W_try.copy())
                    
                    if best_err < 1e-10:
                        break
                
                if best_err < 1e-10:
                    # Found a valid decomposition!
                    best_result = build_decomposition(best_params[0], best_params[1], best_params[2], target_rank)
                    best_rank = target_rank
                    break
            
            if best_rank < 27:
                break
    
    return best_result
