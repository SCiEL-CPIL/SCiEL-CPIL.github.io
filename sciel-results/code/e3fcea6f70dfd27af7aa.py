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
    # Index: a * p + b for a + b*alpha
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    # Point index: x * q + y
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

    # Precompute line membership as a boolean matrix (n_lines x q2)
    line_member = np.zeros((n_lines, q2), dtype=np.int8)
    for li in range(n_lines):
        for pi in line_points[li]:
            line_member[li, pi] = 1

    # S is the set of points in the complement of N
    # We want to maximize |S|
    # Start with S = empty
    S = np.zeros(q2, dtype=bool)
    
    # line_S_count[li] = number of points in S on line li
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    
    # good_line_count[z] = number of lines through z that satisfy the condition for z
    # If z in S: line L is good if line_S_count[L] == 1
    # If z not in S: line L is good if line_S_count[L] == 0
    good_line_count = np.zeros(q2, dtype=np.int64)
    
    # Initialize good_line_count for S = empty
    # For all z, S[z] is False, so we need lines with count 0.
    # All lines have count 0.
    # Number of lines through any point is q + 1 = 50.
    good_line_count[:] = q + 1

    def is_valid():
        return np.all(good_line_count > 0)

    def add_point(z):
        """Try to add z to S. Returns True if successful, False if invalid."""
        # Tentatively add z
        S[z] = True
        
        # Track changes to revert if necessary
        changes = []
        
        # Update line_S_count and good_line_count for lines through z
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            
            # Update good_line_count for all points on this line except z
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    # w in S: good if count == 1
                    if old_count == 1 and new_count != 1:
                        # Was good, now not good
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        # Was not good, now good
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    # w not in S: good if count == 0
                    if old_count == 0 and new_count != 0:
                        # Was good, now not good
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        # Was not good, now good
                        good_line_count[w] += 1
                        changes.append((w, 1))
        
        # Update z's good_line_count
        # z is now in S. Good lines are those with count == 1.
        # The lines through z now have new_count.
        # We need to count how many lines through z have new_count == 1.
        # Note: new_count = old_count + 1. So we need old_count == 0.
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        
        # Check validity
        if is_valid():
            return True
        else:
            # Revert
            for w, delta in changes:
                good_line_count[w] -= delta
            for li in point_lines[z]:
                line_S_count[li] -= 1
            S[z] = False
            return False

    def remove_point(z):
        """Remove z from S. Always valid (removing a point can only help)."""
        if not S[z]:
            return False
        
        S[z] = False
        
        # Update line_S_count and good_line_count for lines through z
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            
            # Update good_line_count for all points on this line except z
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    # w in S: good if count == 1
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                else:
                    # w not in S: good if count == 0
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
        
        # Update z's good_line_count
        # z is now not in S. Good lines are those with count == 0.
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                z_good += 1
        
        good_line_count[z] = z_good
        
        return True

    # Greedy addition first
    rng = np.random.default_rng(42)
    points_to_try = np.arange(q2)
    
    for _pass in range(10):
        if _time_left() <= 0:
            break
        rng.shuffle(points_to_try)
        added_in_pass = 0
        for z in points_to_try:
            if _time_left() <= 0:
                break
            if not S[z]:
                if add_point(z):
                    added_in_pass += 1
        if added_in_pass == 0:
            break

    # Simulated Annealing phase
    # Objective: maximize |S| subject to validity
    # We use a penalty-based approach: if invalid, we penalize
    
    def compute_objective():
        """Return (|S|, -violations) where violations = number of points with good_line_count == 0"""
        s_size = np.sum(S)
        violations = np.sum(good_line_count == 0)
        return s_size, -violations

    best_s_size, _ = compute_objective()
    best_S = S.copy()
    best_line_S_count = line_S_count.copy()
    best_good_line_count = good_line_count.copy()

    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    max_iter = 500000

    rng_sa = np.random.default_rng(123)

    for it in range(max_iter):
        if _time_left() <= 0:
            break
        if T < min_T:
            break

        # Pick a random point to flip
        z = rng_sa.integers(0, q2)

        if S[z]:
            # Try to remove z
            # Removal is always valid, so we just do it
            remove_point(z)
            new_s_size = np.sum(S)
            new_violations = np.sum(good_line_count == 0)
            
            old_s_size = new_s_size + 1
            old_violations = new_violations  # Removal can only decrease or keep violations same
            
            delta_obj = (new_s_size - old_s_size)  # -1
            # We accept if it helps or with some probability
            # But removing reduces |S|, so we only accept if it enables future additions
            # For now, accept with probability based on T
            # Actually, let's just accept removals that reduce violations
            if new_violations < old_violations:
                pass  # Accept
            else:
                # Revert
                add_point(z)
        else:
            # Try to add z
            if add_point(z):
                new_s_size = np.sum(S)
                new_violations = np.sum(good_line_count == 0)
                
                # Check if this is a new best
                if new_s_size > best_s_size:
                    best_s_size = new_s_size
                    best_S = S.copy()
                    best_line_S_count = line_S_count.copy()
                    best_good_line_count = good_line_count.copy()
            else:
                pass  # Reverted already

        T *= cooling

    # Restore best state
    S[:] = best_S
    line_S_count[:] = best_line_S_count
    good_line_count[:] = best_good_line_count

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
