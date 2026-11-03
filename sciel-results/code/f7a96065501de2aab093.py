import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    def schoolbook():
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

    def compute_loss(U_flat, V_flat, W_flat, r):
        U3 = U_flat.reshape(n, m, r)
        V3 = V_flat.reshape(m, p, r)
        W3 = W_flat.reshape(p, n, r)
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        diff = S - 1.0
        return np.sum(diff * diff)

    def verify(U_list, V_list, W_list):
        r = len(U_list[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U_list[i*m+j][r_idx] * V_list[j*p+k][r_idx] * W_list[k*n+i][r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_half_integer(x):
        return round(x * 2) / 2

    def project_to_half_integers(U_flat, V_flat, W_flat, r):
        U3 = U_flat.reshape(n, m, r).copy()
        V3 = V_flat.reshape(m, p, r).copy()
        W3 = W_flat.reshape(p, n, r).copy()
        U3 = np.round(U3 * 2) / 2
        V3 = np.round(V3 * 2) / 2
        W3 = np.round(W3 * 2) / 2
        return U3.flatten(), V3.flatten(), W3.flatten()

    # Search for minimal rank
    # Try ranks from 30 to 35
    best_result = None
    
    for target_rank in range(30, 36):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            strategy = restart % 3
            
            if strategy == 0:
                # Random small values
                vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                U_flat = np.random.choice(vals, size=(n * m * target_rank))
                V_flat = np.random.choice(vals, size=(m * p * target_rank))
                W_flat = np.random.choice(vals, size=(p * n * target_rank))
            elif strategy == 1:
                # Sparse initialization: mostly zeros with some ones
                U_flat = np.zeros(n * m * target_rank)
                V_flat = np.zeros(m * p * target_rank)
                W_flat = np.zeros(p * n * target_rank)
                # Set random positions to 1
                for _ in range(target_rank // 2):
                    U_flat[random.randint(0, n*m*target_rank-1)] = 1
                    V_flat[random.randint(0, m*p*target_rank-1)] = 1
                    W_flat[random.randint(0, p*n*target_rank-1)] = 1
            else:
                # Structured: try to mimic schoolbook but with fewer terms
                U_flat = np.zeros(n * m * target_rank)
                V_flat = np.zeros(m * p * target_rank)
                W_flat = np.zeros(p * n * target_rank)
                # Fill first min(target_rank, 36) positions like schoolbook
                num_terms = min(target_rank, 36)
                for t in range(num_terms):
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U_flat[i*m + j, t] = 1
                    V_flat[j*p + k, t] = 1
                    W_flat[k*n + i, t] = 1
            
            current_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
            best_loss = current_loss
            best_U = U_flat.copy()
            best_V = V_flat.copy()
            best_W = W_flat.copy()
            
            # Local search with simulated annealing
            steps = 1500
            temp = 0.5
            cooling = 0.995
            vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            
            for step in range(steps):
                if time.time() > deadline - 2:
                    break
                
                var_type = random.randint(0, 2)
                if var_type == 0:
                    size = n * m * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = U_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    U_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                elif var_type == 1:
                    size = m * p * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = V_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    V_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                else:
                    size = p * n * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = W_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    W_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U = U_flat.copy()
                    best_V = V_flat.copy()
                    best_W = W_flat.copy()
                    
                    if best_loss < 1e-9:
                        U_list = [[best_U[(i*m+j)*target_rank + r] for r in range(target_rank)] for i in range(n) for j in range(m)]
                        V_list = [[best_V[(j*p+k)*target_rank + r] for r in range(target_rank)] for j in range(m) for k in range(p)]
                        W_list = [[best_W[(k*n+i)*target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                
                if new_loss <= current_loss:
                    current_loss = new_loss
                else:
                    if var_type == 0:
                        U_flat[idx] = old_val
                    elif var_type == 1:
                        V_flat[idx] = old_val
                    else:
                        W_flat[idx] = old_val
                    
                    if random.random() < np.exp((current_loss - new_loss) / temp):
                        if var_type == 0:
                            U_flat[idx] = new_val
                        elif var_type == 1:
                            V_flat[idx] = new_val
                        else:
                            W_flat[idx] = new_val
                        current_loss = new_loss
                
                temp *= cooling
                if temp < 0.01:
                    break
            
            if found:
                break
    
    return schoolbook()
# EVOLVE-BLOCK-END
