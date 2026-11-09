import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Build the matrix multiplication tensor over the rationals
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) with matching indices
    T = np.zeros((size_a, size_b, size_c), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Strategy: Randomized greedy rank-1 peeling over rationals
    # At each step, try to find a rank-1 tensor u ⊗ v ⊗ w that
    # covers as many remaining 1s as possible (ideally multiple entries)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    T_curr = T.copy()
    
    start_time = time.time()
    max_iter = 104  # We want to find rank < 105
    
    # Set random seed for reproducibility within a run
    random.seed(42)
    np.random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() - start_time > 115:
            break
            
        # Check if tensor is zero
        nonzero = np.nonzero(T_curr)
        if len(nonzero[0]) == 0:
            break
        
        # Get all nonzero positions
        positions = list(zip(nonzero[0], nonzero[1], nonzero[2]))
        
        if len(positions) == 0:
            break
        
        # Try to find a good rank-1 term
        # A rank-1 term u ⊗ v ⊗ w covers position (a,b,c) if u[a]*v[b]*w[c] != 0
        # and the sum of contributions matches T_curr[a,b,c]
        
        # Heuristic: Pick a random subset of positions and try to find
        # u, v, w that cover them all with value 1
        
        # For simplicity, try to cover positions that share common indices
        # Group positions by their a-index
        from collections import defaultdict
        by_a = defaultdict(list)
        by_b = defaultdict(list)
        by_c = defaultdict(list)
        for pos in positions:
            by_a[pos[0]].append(pos)
            by_b[pos[1]].append(pos)
            by_c[pos[2]].append(pos)
        
        # Try each a-index as anchor
        best_u = None
        best_v = None
        best_w = None
        best_covered = set()
        best_val = 0
        
        # Try a few random anchors
        anchor_candidates = list(by_a.keys())
        random.shuffle(anchor_candidates)
        
        for a_anchor in anchor_candidates[:10]:  # Try up to 10 anchors
            # Get all positions with this a
            pos_a = by_a[a_anchor]
            
            # For these positions, b and c vary
            # u[a_anchor] = 1 (set to 1 for simplicity)
            # We need v[b]*w[c] = 1 for each (b,c) in pos_a
            
            # Collect unique b's and c's
            b_set = set(pos[1] for pos in pos_a)
            c_set = set(pos[2] for pos in pos_a)
            
            # Try to assign v and w values
            # v[b]*w[c] = 1 for all (b,c) in pos_a
            # This means v[b] = 1/w[c] for all pairs
            # So all v[b] must be equal (if multiple c's) and all w[c] must be equal
            
            # Check consistency
            v_vals = {}
            w_vals = {}
            consistent = True
            
            for pos in pos_a:
                b, c = pos[1], pos[2]
                if b in v_vals:
                    if c in w_vals:
                        if v_vals[b] * w_vals[c] != 1:
                            consistent = False
                            break
                    else:
                        w_vals[c] = 1 / v_vals[b]
                else:
                    if c in w_vals:
                        v_vals[b] = 1 / w_vals[c]
                    else:
                        v_vals[b] = 1
                        w_vals[c] = 1
            
            if not consistent:
                continue
            
            # Now check if this assignment covers other positions too
            # u has 1 at a_anchor, 0 elsewhere for now
            # v has values at b_set, 0 elsewhere
            # w has values at c_set, 0 elsewhere
            
            u = [0] * size_a
            v = [0] * size_b
            w = [0] * size_c
            u[a_anchor] = 1
            for b, val in v_vals.items():
                v[b] = val
            for c, val in w_vals.items():
                w[c] = val
            
            # Count how many positions are covered
            covered = set()
            for pos in positions:
                a, b, c = pos
                if u[a] != 0 and v[b] != 0 and w[c] != 0:
                    if u[a] * v[b] * w[c] == 1:
                        covered.add(pos)
            
            if len(covered) > best_val:
                best_val = len(covered)
                best_u = u[:]
                best_v = v[:]
                best_w = w[:]
                best_covered = covered
        
        if best_val == 0:
            # Fallback: use schoolbook step
            a_idx, b_idx, c_idx = positions[0]
            u = [0] * size_a
            v = [0] * size_b
            w = [0] * size_c
            u[a_idx] = 1
            v[b_idx] = 1
            w[c_idx] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            T_curr[a_idx, b_idx, c_idx] = 0
            continue
        
        # Add the best rank-1 term
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Subtract the contribution from T_curr
        for pos in best_covered:
            a, b, c = pos
            T_curr[a, b, c] = T_curr[a, b, c] - best_u[a] * best_v[b] * best_w[c]
            if abs(T_curr[a, b, c]) < 1e-9:
                T_curr[a, b, c] = 0
    
    # Check if we have a valid decomposition
    if len(U_cols) < 105:
        # Verify the decomposition
        rank = len(U_cols)
        U = [[0] * rank for _ in range(size_a)]
        V = [[0] * rank for _ in range(size_b)]
        W = [[0] * rank for _ in range(size_c)]
        for r in range(rank):
            for a in range(size_a):
                U[a][r] = U_cols[r][a]
            for b in range(size_b):
                V[b][r] = V_cols[r][b]
            for c in range(size_c):
                W[c][r] = W_cols[r][c]
        
        # Verify: sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
        # Quick check on a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(size_a)]
    V = [[0] * rank for _ in range(size_b)]
    W = [[0] * rank for _ in range(size_c)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
