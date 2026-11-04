import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for search
    
    n, m, p = DIMENSIONS
    
    # Try to find a low rank decomposition for 2x3x3
    # Then replicate for 2x3x9
    best_rank_233 = None
    best_U_233 = None
    best_V_233 = None
    best_W_233 = None
    
    # Target tensor for 2x3x3
    # Indices: i in [0,1], j in [0,2], k in [0,2]
    # U size: 2*3 = 6
    # V size: 3*3 = 9
    # W size: 3*2 = 6
    
    target_U = np.zeros((6, 1)) # Placeholder
    # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    # We define the residual R_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} - 1
    
    def check_solution(U, V, W):
        # U: (6, R), V: (9, R), W: (6, R)
        # Check if U[i,j] V[j,k] W[k,i] sums to 1
        # i: 0..1, j: 0..2, k: 0..2
        # U index: i*3 + j
        # V index: j*3 + k
        # W index: k*2 + i
        max_err = 0.0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    ui = i * 3 + j
                    vk = j * 3 + k
                    wk = k * 2 + i
                    s = np.sum(U[ui, :] * V[vk, :] * W[wk, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def search_rank(R):
        # Random initialization
        U = np.random.uniform(-2, 2, (6, R))
        V = np.random.uniform(-2, 2, (9, R))
        W = np.random.uniform(-2, 2, (6, R))
        
        # Local search / Gradient Descent
        # Since it's discrete, we can try a few random restarts
        for restart in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.uniform(-2, 2, (6, R))
            V = np.random.uniform(-2, 2, (9, R))
            W = np.random.uniform(-2, 2, (6, R))
            
            # Simple local search: perturb and round
            # Or just check random integers in a small range?
            # Given the small size, maybe random search over small integers works?
            # Try integers -2, -1, 0, 1, 2
            # But R is up to 16, so 5^16 is huge.
            
            # Let's use a simple gradient step on real values, then round.
            for step in range(20):
                if time.time() > deadline:
                    break
                # Calculate residual
                # This is expensive if done in Python loops.
                # But 18 terms is small.
                # Let's just do a few random perturbations.
                
                # Pick a random element to perturb
                mat = random.choice([U, V, W])
                r, c = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
                old_val = mat[r, c]
                mat[r, c] += random.uniform(-0.5, 0.5)
                
                # Check if error decreased? 
                # This is too slow to calculate full error every time.
                
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            err = check_solution(U_r, V_r, W_r)
            if err < 1e-6:
                return U_r, V_r, W_r
        return None

    # Try ranks from low to high
    for R in range(12, 18):
        res = search_rank(R)
        if res is not None:
            best_rank_233 = R
            best_U_233, best_V_233, best_W_233 = res
            break
        if time.time() > deadline:
            break

    if best_rank_233 is not None:
        # Construct 2x3x9 decomposition
        # U: (2*3, 3*R_233) = (6, 3R)
        # V: (3*9, 3*R_233) = (27, 3R)
        # W: (9*2, 3*R_233) = (18, 3R)
        
        R_total = 3 * best_rank_233
        U = np.zeros((6, R_total))
        V = np.zeros((27, R_total))
        W = np.zeros((18, R_total))
        
        for s in range(3):
            U[:, s*best_rank_233:(s+1)*best_rank_233] = best_U_233
            # V block s: rows j*9 + (3s + k')?
            # V index is j*p + k. p=9.
            # For block s, k = 3s + k'.
            # So V[j*9 + (3s+k'), s*R : (s+1)*R] = best_V_233[j*3 + k', :]
            for j in range(3):
                for k_prime in range(3):
                    V[j*9 + (3*s + k_prime), s*best_rank_233:(s+1)*best_rank_233] = best_V_233[j*3 + k_prime, :]
            
            # W block s: rows k*2 + i.
            # For block s, k = 3s + k'.
            # So W[(3s+k')*2 + i, s*R : (s+1)*R] = best_W_233[k'*2 + i, :]
            for k_prime in range(3):
                for i in range(2):
                    W[(3*s + k_prime)*2 + i, s*best_rank_233:(s+1)*best_rank_233] = best_W_233[k_prime*2 + i, :]
        
        # Convert to lists for output
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
