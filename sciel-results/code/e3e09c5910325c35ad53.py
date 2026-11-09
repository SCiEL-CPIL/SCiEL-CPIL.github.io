import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 7
    nm, mp, np_ = n * m, m * p, p * n  # 6, 21, 14
    deadline = time.time() + 140
    
    # Schoolbook fallback (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Target support
    target_support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_support.add((u, v, w))
    
    def compute_error(U, V, W, r):
        """Compute sum of squared errors over all (u,v,w) tuples."""
        err = 0.0
        for u in range(nm):
            for v in range(mp):
                for w in range(np_):
                    s = 0.0
                    for c in range(r):
                        s += U[u, c] * V[v, c] * W[w, c]
                    target = 1.0 if (u, v, w) in target_support else 0.0
                    err += (s - target) ** 2
        return err
    
    def is_valid(U, V, W, r):
        """Check if decomposition is exact."""
        for (u, v, w) in target_support:
            s = 0.0
            for c in range(r):
                s += U[u, c] * V[v, c] * W[w, c]
            if abs(s - 1.0) > 1e-9:
                return False
        # Check non-support entries
        for u in range(nm):
            for v in range(mp):
                for w in range(np_):
                    if (u, v, w) in target_support:
                        continue
                    s = 0.0
                    for c in range(r):
                        s += U[u, c] * V[v, c] * W[w, c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Structured initialization: exploit n=2 sparsity in U
    # Each U column should have entries only in rows corresponding to one i-block
    def structured_init(r, values):
        """Create structured random initialization respecting U's block structure."""
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((np_, r))
        
        for c in range(r):
            # U column: pick one i-block (i=0 or i=1) and assign values to m entries
            i_block = random.choice([0, 1])
            for j in range(m):
                U[i_block * m + j, c] = random.choice(values)
            
            # V and W: fully random
            for v in range(mp):
                V[v, c] = random.choice(values)
            for w in range(np_):
                W[w, c] = random.choice(values)
        
        return U, V, W
    
    def local_search(U, V, W, r, values, max_steps=5000):
        """Local search with integer/half-integer perturbations."""
        current_err = compute_error(U, V, W, r)
        if current_err < 1e-9:
            return True, U, V, W
        
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline:
                return False, U, V, W
            
            # Pick random matrix, entry, and new value
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                dim = nm
            elif mat_idx == 1:
                mat = V
                dim = mp
            else:
                mat = W
                dim = np_
            
            idx = random.randrange(dim)
            col = random.randrange(r)
            
            old_val = mat[idx, col]
            new_val = random.choice(values)
            
            if new_val == old_val:
                continue
            
            mat[idx, col] = new_val
            new_err = compute_error(U, V, W, r)
            
            if new_err < current_err - 1e-12:
                current_err = new_err
                stuck = 0
                if current_err < 1e-9:
                    return True, U, V, W
            else:
                mat[idx, col] = old_val
                stuck += 1
            
            if stuck > 200:
                break
        
        return False, U, V, W
    
    # Search from low rank upward
    # Known lower bounds suggest rank >= 19 for this size, but let's try from 25
    for r in range(25, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            values = [0, 1, -1]
            if r <= 30:
                values = [0, 1, -1, 0.5, -0.5]
            
            U, V, W = structured_init(r, values)
            success, U, V, W = local_search(U, V, W, r, values, max_steps=3000)
            
            if success:
                # Verify
                if is_valid(U, V, W, r):
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    # Final verification
    if is_valid(best_U, best_V, best_W, best_rank):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
