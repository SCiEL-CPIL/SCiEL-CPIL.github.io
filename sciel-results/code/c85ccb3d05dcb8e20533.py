import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 parts
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_terms():
        terms = []
        # M0
        terms.append({'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}})
        # M1
        terms.append({'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}})
        # M2
        terms.append({'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}})
        # M3
        terms.append({'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}})
        # M4
        terms.append({'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}})
        # M5
        terms.append({'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}})
        # M6
        terms.append({'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}})
        return terms

    terms = get_strassen_terms()
    
    col = 0
    for j_high in range(2):
        for l in range(4):
            for t_idx in range(7):
                term = terms[t_idx]
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (k_low, i), coeff in term['W'].items():
                    k = l * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
                
    # Convert to numpy arrays for efficient computation
    U_np = np.array(U, dtype=np.int64)  # shape (n*m, rank)
    V_np = np.array(V, dtype=np.int64)  # shape (m*p, rank)
    W_np = np.array(W, dtype=np.int64)  # shape (p*n, rank)
    
    # Precompute the contribution of each column to the tensor
    # For each column c, compute the outer product U[:,c] * V[:,c] * W[:,c] reshaped to (n,m,p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def compute_column_contribution(c):
        """Compute the (n,m,p) tensor for column c"""
        U_c = U_np[:, c].reshape(n, m)  # (n, m)
        V_c = V_np[:, c].reshape(m, p)  # (m, p)
        W_c = W_np[:, c].reshape(p, n)  # (p, n)
        
        # T[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        # Use broadcasting
        result = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
        return result
    
    # Precompute contributions for all columns
    col_contributions = []
    for c in range(rank):
        col_contributions.append(compute_column_contribution(c))
    
    # Target tensor: all ones
    target = np.ones((n, m, p), dtype=np.int64)
    
    # Greedy column removal with vectorized checking
    active = list(range(rank))
    deadline = time.time() + 140
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing each active column
        for c in list(active):
            if time.time() > deadline:
                break
            # Check if removing column c still gives the target tensor
            # Sum of all active contributions minus column c should equal target
            new_active = [x for x in active if x != c]
            
            # Compute sum of contributions for new_active
            total = np.zeros((n, m, p), dtype=np.int64)
            for ac in new_active:
                total += col_contributions[ac]
            
            if np.array_equal(total, target):
                active = new_active
                changed = True
                break  # Restart to try removing another
        
        # If no column was removed in this pass, try a different order
        if not changed:
            # Try random order
            random_order = active.copy()
            np.random.shuffle(random_order)
            for c in random_order:
                if time.time() > deadline:
                    break
                new_active = [x for x in active if x != c]
                
                total = np.zeros((n, m, p), dtype=np.int64)
                for ac in new_active:
                    total += col_contributions[ac]
                
                if np.array_equal(total, target):
                    active = new_active
                    changed = True
                    break
    
    # Reconstruct matrices with only active columns
    if len(active) < rank:
        U_final = [[U[i][c] for c in active] for i in range(n*m)]
        V_final = [[V[i][c] for c in active] for i in range(m*p)]
        W_final = [[W[i][c] for c in active] for i in range(p*n)]
        return {"U": U_final, "V": V_final, "W": W_final}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
