# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # State: S is the complement of N
    # S is valid if:
    # - for z in S: exists line L through z with count_S(L) == 1
    # - for z not in S: exists line L through z with count_S(L) == 0
    
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    good_line_count = np.zeros(q2, dtype=np.int64)
    
    # Initialize with S = parabola y = x^2 (size 49)
    # This is a known valid Nikodym set complement
    for x in range(q):
        y = int(mul_table[x, x])
        z = pt_idx(x, y)
        S[z] = True
        for li in point_lines[z]:
            line_S_count[li] += 1
            
    # Calculate good_line_count for initial state
    for z in range(q2):
        if S[z]:
            for li in point_lines[z]:
                if line_S_count[li] == 1:
                    good_line_count[z] += 1
        else:
            for li in point_lines[z]:
                if line_S_count[li] == 0:
                    good_line_count[z] += 1
    
    best_S = S.copy()
    best_size = np.sum(S)
    
    def try_add(z):
        """Try to add z to S. Returns True if valid."""
        if S[z]:
            return False
        # Check if adding z keeps validity
        S[z] = True
        affected = set()
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            
            for w in line_points[li]:
                if w == z:
                    continue
                affected.add(w)
                if S[w]:
                    if old_count == 1:
                        good_line_count[w] -= 1
                    elif new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0:
                        good_line_count[w] -= 1
                    elif new_count == 0:
                        good_line_count[w] += 1
        
        # Update z
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        good_line_count[z] = z_good
        
        if np.all(good_line_count >= 1):
            return True
        else:
            # Revert
            S[z] = False
            for li in point_lines[z]:
                old_count = line_S_count[li]
                new_count = old_count - 1
                line_S_count[li] = new_count
                
                for w in line_points[li]:
                    if w == z:
                        continue
                    if S[w]:
                        if new_count == 1:
                            good_line_count[w] += 1
                        elif old_count == 1:
                            good_line_count[w] -= 1
                    else:
                        if new_count == 0:
                            good_line_count[w] += 1
                        elif old_count == 0:
                            good_line_count[w] -= 1
            
            z_good = 0
            for li in point_lines[z]:
                if line_S_count[li] == 0:
                    z_good += 1
            good_line_count[z] = z_good
            return False

    def try_swap(u, v):
        """Try to swap u (in S) with v (not in S). Returns True if valid."""
        if not S[u] or S[v]:
            return False
        # Remove u
        S[u] = False
        for li in point_lines[u]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            for w in line_points[li]:
                if w == u:
                    continue
                if S[w]:
                    if old_count == 1:
                        good_line_count[w] -= 1
                    elif new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0:
                        good_line_count[w] -= 1
                    elif new_count == 0:
                        good_line_count[w] += 1
        # Update u
        u_good = 0
        for li in point_lines[u]:
            if line_S_count[li] == 0:
                u_good += 1
        good_line_count[u] = u_good
        
        # Add v
        S[v] = True
        for li in point_lines[v]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            for w in line_points[li]:
                if w == v:
                    continue
                if S[w]:
                    if old_count == 1:
                        good_line_count[w] -= 1
                    elif new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0:
                        good_line_count[w] -= 1
                    elif new_count == 0:
                        good_line_count[w] += 1
        # Update v
        v_good = 0
        for li in point_lines[v]:
            if line_S_count[li] == 1:
                v_good += 1
        good_line_count[v] = v_good
        
        if np.all(good_line_count >= 1):
            return True
        else:
            # Revert Add v
            S[v] = False
            for li in point_lines[v]:
                old_count = line_S_count[li]
                new_count = old_count - 1
                line_S_count[li] = new_count
                for w in line_points[li]:
                    if w == v:
                        continue
                    if S[w]:
                        if new_count == 1:
                            good_line_count[w] += 1
                        elif old_count == 1:
                            good_line_count[w] -= 1
                    else:
                        if new_count == 0:
                            good_line_count[w] += 1
                        elif old_count == 0:
                            good_line_count[w] -= 1
            v_good = 0
            for li in point_lines[v]:
                if line_S_count[li] == 0:
                    v_good += 1
            good_line_count[v] = v_good
            
            # Revert Remove u
            S[u] = True
            for li in point_lines[u]:
                old_count = line_S_count[li]
                new_count = old_count + 1
                line_S_count[li] = new_count
                for w in line_points[li]:
                    if w == u:
                        continue
                    if S[w]:
                        if old_count == 1:
                            good_line_count[w] -= 1
                        elif new_count == 1:
                            good_line_count[w] += 1
                    else:
                        if old_count == 0:
                            good_line_count[w] -= 1
                        elif new_count == 0:
                            good_line_count[w] += 1
            u_good = 0
            for li in point_lines[u]:
                if line_S_count[li] == 1:
                    u_good += 1
            good_line_count[u] = u_good
            return False

    rng = np.random.default_rng(42)
    
    # Simulated Annealing / Local Search
    # We want to maximize |S|.
    # Start with S = Parabola (size 49).
    # Try to Add points. If fail, try Swaps to change structure.
    
    current_size = np.sum(S)
    
    while _time_left() > 10:
        # Try to add a random point
        candidates = np.where(~S)[0]
        if len(candidates) > 0:
            v = candidates[rng.integers(0, len(candidates))]
            if try_add(v):
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
                    # Restart or continue? Continue.
        
        # Try a swap
        u_candidates = np.where(S)[0]
        v_candidates = np.where(~S)[0]
        if len(u_candidates) > 0 and len(v_candidates) > 0:
            u = u_candidates[rng.integers(0, len(u_candidates))]
            v = v_candidates[rng.integers(0, len(v_candidates))]
            if try_swap(u, v):
                # Size unchanged, but structure changed
                pass
        
        # Occasionally restart from a random line if stuck?
        # For now, just keep going.
        
    # Build output
    pts_list = []
    for z in range(q2):
        if not best_S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
