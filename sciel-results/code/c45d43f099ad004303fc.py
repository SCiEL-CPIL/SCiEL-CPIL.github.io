import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
            
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    # Convert line_points to lists for faster iteration
    line_points_list = [list(s) for s in line_points]
    
    # For each line, precompute the set of points on that line (excluding self for witness check)
    # line_witness_check[lid] = list of (point_idx) on that line
    
    def build_W(in_C):
        """Build witness sets W[i] = set of line IDs where all other points on line are in C."""
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and not in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        return W
    
    def count_witnesss(W):
        """Count total witnesses across all points."""
        return sum(len(w) for w in W)
    
    def try_add_point(p_idx, in_C, W):
        """Try to add p_idx to C. Returns True if successful."""
        # Check: for each line through p_idx, after adding p_idx to C,
        # does every point on that line still have at least one witness?
        # A point x on line lid (x != p_idx) loses lid as witness if lid was its only witness.
        affected = set()
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        if len(W[x_idx]) == 1:
                            # This point would lose its only witness
                            return False
                        else:
                            affected.add(x_idx)
        
        # It's safe to add
        in_C[p_idx] = True
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
        return True
    
    def try_remove_point(p_idx, in_C, W):
        """Try to remove p_idx from C. Returns True if successful."""
        # After removing p_idx from C, for each line through p_idx,
        # points on that line that had lid as witness may lose it.
        # But actually, removing from C means lid is NO LONGER a valid witness for anyone.
        # So we need to check: for each point x on a line through p_idx,
        # if lid is in W[x], it must be removed, and x must still have >= 1 witness.
        
        affected = set()
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        if len(W[x_idx]) == 1:
                            # This point would lose its only witness
                            return False
                        else:
                            affected.add(x_idx)
        
        # It's safe to remove
        in_C[p_idx] = False
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
        return True
    
    best_in_C = None
    best_size = -1
    
    rng = random.Random(42)
    
    # Strategy: Try multiple initialization strategies
    # 1. Parabola
    # 2. Random large sets
    # 3. Union of shifted parabolas
    
    inits = []
    
    # Init 1: Parabola y = x^2 in F_49
    in_C = np.zeros(q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            pi = x_idx * q + y_idx
            in_C[pi] = True
    inits.append(in_C.copy())
    
    # Init 2: Larger random set (try to start with ~500 points)
    for trial in range(10):
        in_C = np.zeros(q2, dtype=bool)
        # Randomly select ~400-600 points
        n_points = rng.randint(400, 600)
        indices = rng.sample(range(q2), n_points)
        for idx in indices:
            in_C[idx] = True
        inits.append(in_C.copy())
    
    # Init 3: Union of shifted parabolas
    for shift in range(5):
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb + shift) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        inits.append(in_C.copy())
    
    for init_idx, init_in_C in enumerate(inits):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        in_C = init_in_C.copy()
        W = build_W(in_C)
        
        # Check validity: every point must have at least one witness
        valid = True
        for i in range(q2):
            if len(W[i]) == 0:
                valid = False
                break
        
        if not valid:
            # Try to fix by removing problematic points from C
            # A point p in C is "problematic" if removing it doesn't help (still some point has 0 witnesses)
            # Actually, if a point has 0 witnesses, it means for every line through it, there's some other point not in C.
            # We can try to add those missing points to C.
            # For now, skip this init if invalid.
            continue
        
        # Local search: iteratively try to add points to C
        # Use a queue of candidate points
        candidates = [i for i in range(q2) if not in_C[i]]
        rng.shuffle(candidates)
        
        improved = True
        while improved and time.time() < _DEADLINE - _MARGIN:
            improved = False
            for p_idx in candidates:
                if time.time() > _DEADLINE - _MARGIN:
                    break
                if in_C[p_idx]:
                    continue
                if try_add_point(p_idx, in_C, W):
                    improved = True
                    # Remove from candidates
                    candidates = [c for c in candidates if c != p_idx]
                    # Add neighbors to candidates (points on lines through p_idx)
                    for lid in point_lines[p_idx]:
                        for x_idx in line_points_list[lid]:
                            if x_idx != p_idx and not in_C[x_idx] and x_idx not in candidates:
                                candidates.append(x_idx)
        
        # Also try removing points from C to see if we can get a smaller C (larger N)
        # Wait - we want LARGER C for smaller N. So we only want to add to C.
        # But maybe some points in C are "redundant" and can be removed?
        # No - removing from C makes C smaller, which makes N larger. That's bad.
        # We want to maximize |C|, so we only add.
        
        current_size = int(np.sum(in_C))
        if current_size > best_size:
            best_size = current_size
            best_in_C = in_C.copy()
    
    # If no valid init was found, fall back to parabola
    if best_in_C is None:
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        best_in_C = in_C
        best_size = int(np.sum(in_C))
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)
