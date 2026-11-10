import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_pts = q * q
    n_dirs = q + 1
    
    # Directions:
    # 0..q-1: (1, m) for m in F_q
    # q: (0, 1)
    
    # Precompute mapping from point index to (xa, xb, ya, yb)
    # Index = xa * q + xb * p + ya? No.
    # Point is [[xa, xb], [ya, yb]].
    # Let's use idx = ((xa * p + xb) * p + ya) * p + yb?
    # Simpler: idx = xa * q + xb * p + ya?
    # Let's map (xa, xb, ya, yb) to idx.
    # idx = (xa * p + xb) * q + (ya * p + yb) ?
    # Let's stick to:
    # x = xa * p + xb (value in F_q represented as int 0..48? No, pair)
    # We need field arithmetic.
    
    # Field F_49 = F_7[alpha], alpha^2 = 3.
    # Add: (a,b) + (c,d) = ((a+c)%7, (b+d)%7)
    # Mul: (a,b) * (c,d) = (a*c + 3*b*d, a*d + b*c) mod 7
    
    def to_val(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
        
    def from_val(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Precompute direction index from p to q
    # dir_idx[p_idx][q_idx] = v_idx
    # If p == q, v_idx = -1
    dir_idx = np.full((n_pts, n_pts), -1, dtype=np.int8)
    
    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append((xa, xb, ya, yb))
                    
    # Map index to point
    # idx = to_val(...)
    
    # Compute directions
    # v = (1, m) or (0, 1)
    # m is in F_q. Represent m as (ma, mb).
    # Line through x with dir v: { x + t*v }
    # y = x + t*v => y - x = t*v.
    # If v=(0,1), y_x - x_x = 0, y_y - x_y = t.
    #   So y_xa == x_xa, y_xb == x_xb.
    #   dir_idx is q (the (0,1) direction).
    # If v=(1, m), y_x - x_x = t, y_y - x_y = t*m.
    #   If y_x == x_x, then t=0 => y==x.
    #   If y_x != x_x, t = y_x - x_x.
    #   Check if y_y - x_y == t*m.
    
    for i in range(n_pts):
        xa, xb, ya, yb = from_val(i)
        for j in range(n_pts):
            if i == j: continue
            x2a, x2b, y2a, y2b = from_val(j)
            
            # Check (0,1) direction
            if xa == x2a and xb == x2b:
                dir_idx[i, j] = q
                continue
            
            # Check (1, m) directions
            # t = y2_x - y_x
            # t_a = (x2a - xa) % p
            # t_b = (x2b - xb) % p
            ta = (x2a - xa) % p
            tb = (x2b - xb) % p
            
            # m = t_y / t_x
            # We need m such that t_y = m * t_x
            # If t_x == 0, no solution (already handled by (0,1) or i==j)
            # Here t_x != 0 because xa!=x2a or xb!=x2b is not sufficient?
            # Wait, (0,1) covers x_x same.
            # So here t_x != 0.
            
            # Invert t_x in F_49
            # t_x = (ta, tb). Inverse is (ta, -tb) / (ta^2 - 3 tb^2)?
            # Norm = ta^2 - 3 tb^2.
            # If Norm == 0, t_x = 0. But we said t_x != 0.
            # Inverse = conjugate / Norm.
            norm = (ta * ta - 3 * tb * tb) % p
            if norm == 0:
                # Should not happen if t_x != 0
                continue
            norm_inv = pow(norm, p - 2, p) # Fermat
            t_inv_a = ta * norm_inv % p
            t_inv_b = -tb * norm_inv % p
            
            # m = t_y * t_inv
            # t_y = (y2a - ya, y2b - yb)
            ty_a = (y2a - ya) % p
            ty_b = (y2b - yb) % p
            
            ma = (ty_a * t_inv_a + 3 * ty_b * t_inv_b) % p
            mb = (ty_a * t_inv_b + ty_b * t_inv_a) % p
            
            # m is (ma, mb). Map to index 0..q-1.
            m_idx = ma * p + mb
            dir_idx[i, j] = m_idx

    def run_greedy(seed):
        random.seed(seed)
        S = [False] * n_pts
        # W_N[i] bitmask of directions where L(i, v) cap S is empty
        # W_S[i] bitmask of directions where L(i, v) cap S is {i}
        W_N = [ (1 << n_dirs) - 1 ] * n_pts
        W_S = [ 0 ] * n_pts
        
        order = list(range(n_pts))
        random.shuffle(order)
        
        best_S_size = 0
        best_S = S[:]
        
        for p_idx in order:
            # Check if p_idx can be added
            if W_N[p_idx] == 0:
                continue
            
            # Save state
            old_W_N = W_N[:]
            old_W_S = W_S[:]
            old_S = S[:]
            
            # Tentatively add p_idx
            S[p_idx] = True
            W_S[p_idx] = W_N[p_idx] # The directions that were empty are now isolating
            W_N[p_idx] = 0
            
            valid = True
            # Check all other points
            for q_idx in range(n_pts):
                if q_idx == p_idx: continue
                
                v = dir_idx[p_idx, q_idx]
                if v == -1: continue
                mask = 1 << v
                
                if old_S[q_idx]: # q was in S
                    if old_W_S[q_idx] & mask:
                        new_ws = old_W_S[q_idx] & ~mask
                        W_S[q_idx] = new_ws
                        if new_ws == 0:
                            valid = False
                            break
                else: # q was in N
                    if old_W_N[q_idx] & mask:
                        new_wn = old_W_N[q_idx] & ~mask
                        W_N[q_idx] = new_wn
                        if new_wn == 0:
                            valid = False
                            break
            
            if valid:
                if sum(S) > best_S_size:
                    best_S_size = sum(S)
                    best_S = S[:]
            else:
                S = old_S
                W_N = old_W_N
                W_S = old_W_S
                
        # Construct result
        # Return N = complement of S
        res = []
        for i in range(n_pts):
            if not best_S[i]:
                xa, xb, ya, yb = from_val(i)
                res.append([[xa, xb], [ya, yb]])
        return np.array(res, dtype=np.int64)

    # Run with a few seeds
    start_time = time.time()
    timeout = 600
    
    best_result = None
    best_size = -1
    
    # Try a few seeds
    for seed in range(10):
        if time.time() - start_time > timeout - 10:
            break
        res = run_greedy(seed)
        size = len(res)
        comp_size = 2401 - size
        if comp_size > best_size:
            best_size = comp_size
            best_result = res
            
    if best_result is None:
        # Fallback to baseline
        return np.array([[[xa, xb], [ya, yb]] for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)], dtype=np.int64)
        
    return best_result
