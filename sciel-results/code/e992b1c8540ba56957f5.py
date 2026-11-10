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
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    good_line_count = np.zeros(q2, dtype=np.int64)

    # Initialize with S = parabola y = x^2 (size 49)
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
    best_size = int(np.sum(S))
    best_line_S_count = line_S_count.copy()
    best_good_line_count = good_line_count.copy()

    def try_add(z):
        """Try to add z to S. Returns True if valid."""
        if S[z]:
            return False
        S[z] = True
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            for w in line_points[li]:
                if w == z:
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
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        good_line_count[z] = z_good
        if np.all(good_line_count >= 1):
            return True
        else:
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

    # Phase 1: Greedy addition from parabola
    rng = np.random.default_rng(42)
    points_to_try = np.arange(q2)

    for _pass in range(20):
        if _time_left() <= 0:
            break
        rng.shuffle(points_to_try)
        added_in_pass = 0
        for z in points_to_try:
            if _time_left() <= 0:
                break
            if not S[z]:
                if try_add(z):
                    added_in_pass += 1
                    cur_size = int(np.sum(S))
                    if cur_size > best_size:
                        best_size = cur_size
                        best_S = S.copy()
                        best_line_S_count = line_S_count.copy()
                        best_good_line_count = good_line_count.copy()
        if added_in_pass == 0:
            break

    # Phase 2: Swap-based local search with simulated annealing
    T = 5.0
    cooling = 0.99995
    min_T = 0.01
    max_iter = 500000
    rng_sa = np.random.default_rng(123)

    s_in = np.where(S)[0]
    s_not = np.where(~S)[0]

    for it in range(max_iter):
        if _time_left() <= 0:
            break
        if T < min_T:
            break

        move_type = rng_sa.integers(0, 2)
        if move_type == 0:
            # Try add
            if len(s_not) == 0:
                T *= cooling
                continue
            z = s_not[rng_sa.integers(0, len(s_not))]
            if try_add(z):
                s_in = np.append(s_in, z)
                s_not = s_not[np.where(s_not != z)]
                cur_size = int(np.sum(S))
                if cur_size > best_size:
                    best_size = cur_size
                    best_S = S.copy()
                    best_line_S_count = line_S_count.copy()
                    best_good_line_count = good_line_count.copy()
            else:
                pass
        else:
            # Try swap
            if len(s_in) == 0 or len(s_not) == 0:
                T *= cooling
                continue
            u = s_in[rng_sa.integers(0, len(s_in))]
            v = s_not[rng_sa.integers(0, len(s_not))]
            if u == v:
                T *= cooling
                continue
            if try_swap(u, v):
                s_in = s_in[s_in != u]
                s_in = np.append(s_in, v)
                s_not = s_not[s_not != v]
                s_not = np.append(s_not, u)
                cur_size = int(np.sum(S))
                if cur_size > best_size:
                    best_size = cur_size
                    best_S = S.copy()
                    best_line_S_count = line_S_count.copy()
                    best_good_line_count = good_line_count.copy()
            else:
                pass

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
