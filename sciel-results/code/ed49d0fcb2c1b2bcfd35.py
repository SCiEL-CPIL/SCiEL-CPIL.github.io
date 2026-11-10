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

    # S is the set of points in the complement of N
    # We want to maximize |S|
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    good_line_count = np.zeros(q2, dtype=np.int64)
    good_line_count[:] = q + 1

    def is_valid():
        return np.all(good_line_count > 0)

    def add_point(z):
        S[z] = True
        changes = []
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        if is_valid():
            return True
        else:
            for w, delta in changes:
                good_line_count[w] -= delta
            for li in point_lines[z]:
                line_S_count[li] -= 1
            S[z] = False
            return False

    def remove_point(z):
        if not S[z]:
            return False
        S[z] = False
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                z_good += 1
        good_line_count[z] = z_good
        return True

    # Phase 1: Greedy initialization
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

    best_s_size = int(np.sum(S))
    best_S = S.copy()
    best_line_S_count = line_S_count.copy()
    best_good_line_count = good_line_count.copy()

    # Phase 2: Simulated Annealing with pair-flip neighborhood
    # A move: add point z1 and remove point z2 (or just add, or just remove)
    # This allows escaping local optima where no single add works but a swap does.
    
    rng_sa = np.random.default_rng(12345)
    T_start = 10.0
    T_end = 0.1
    n_steps = 50000
    
    current_s_size = best_s_size
    S[:] = best_S
    line_S_count[:] = best_line_S_count
    good_line_count[:] = best_good_line_count

    in_S_list = np.where(S)[0].tolist()
    not_in_S_list = np.where(~S)[0].tolist()

    for step in range(n_steps):
        if _time_left() <= 0:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        move_type = rng_sa.random()
        
        if move_type < 0.4:
            # Single add
            if len(not_in_S_list) == 0:
                continue
            z = int(not_in_S_list[rng_sa.integers(0, len(not_in_S_list))])
            if add_point(z):
                not_in_S_list.remove(z)
                in_S_list.append(z)
                current_s_size += 1
                if current_s_size > best_s_size:
                    best_s_size = current_s_size
                    best_S = S.copy()
                    best_line_S_count = line_S_count.copy()
                    best_good_line_count = good_line_count.copy()
            else:
                pass
        elif move_type < 0.6:
            # Single remove
            if len(in_S_list) == 0:
                continue
            z = int(in_S_list[rng_sa.integers(0, len(in_S_list))])
            remove_point(z)
            in_S_list.remove(z)
            not_in_S_list.append(z)
            current_s_size -= 1
        else:
            # Pair flip: add z1, remove z2
            if len(not_in_S_list) == 0 or len(in_S_list) == 0:
                continue
            z1 = int(not_in_S_list[rng_sa.integers(0, len(not_in_S_list))])
            z2 = int(in_S_list[rng_sa.integers(0, len(in_S_list))])
            
            # Try add z1 first
            if add_point(z1):
                # Now try remove z2
                remove_point(z2)
                not_in_S_list.remove(z1)
                in_S_list.append(z1)
                in_S_list.remove(z2)
                not_in_S_list.append(z2)
                # Net change: +1 - 1 = 0 in size, but may improve validity structure
                # Actually this keeps size same. Let's check if it helps future adds.
                # For now, accept if valid (it always is after remove)
                # The benefit is structural: different configuration
            else:
                # Revert: z1 was not added
                pass

    # Phase 3: Final greedy polish
    for _pass in range(20):
        if _time_left() <= 0:
            break
        improved = False
        for z in range(q2):
            if _time_left() <= 0:
                break
            if not S[z]:
                if add_point(z):
                    improved = True
        if not improved:
            break

    if int(np.sum(S)) > best_s_size:
        best_s_size = int(np.sum(S))
        best_S = S.copy()
        best_line_S_count = line_S_count.copy()
        best_good_line_count = good_line_count.copy()

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
