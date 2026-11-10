# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    total_points = q * q  # 2401
    n_dirs = 50  # 49 slopes + vertical

    start_time = time.time()
    timeout = 610  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    def add_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        return (a1 + a2) % p + p * ((b1 + b2) % p)

    def sub_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        return (a1 - a2) % p + p * ((b1 - b2) % p)

    def mul_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + b1 * a2) % p
        return a + p * b

    def inv_f(i):
        if i == 0:
            return 0
        a, b = i % p, i // p
        norm = (a * a - 3 * b * b) % p
        norm_inv = pow(norm, p - 2, p)
        return mul_f(i, norm_inv)

    # Precompute all lines
    line_id = np.zeros((total_points, n_dirs), dtype=np.int32)
    for pt in range(total_points):
        xa = pt // (q * p)
        xb = (pt // q) % p
        x_elem = xa + p * xb
        line_id[pt, 49] = x_elem

    for d in range(49):
        for pt in range(total_points):
            xa = pt // (q * p)
            xb = (pt // q) % p
            ya = (pt // p) % p
            yb = pt % p
            x_elem = xa + p * xb
            y_elem = ya + p * yb
            c = sub_f(y_elem, mul_f(d, x_elem))
            line_id[pt, d] = c

    # Build line_points: for each (d, line_val), collect all points
    line_points = {}
    for d in range(n_dirs):
        for lv in range(q):
            pts = []
            for pt in range(total_points):
                if line_id[pt, d] == lv:
                    pts.append(pt)
            line_points[(d, lv)] = np.array(pts, dtype=np.int32)

    # Precompute other_pts[pt][d] = 48 other point indices
    other_pts = np.zeros((total_points, n_dirs, q - 1), dtype=np.int32)
    for d in range(n_dirs):
        for lv in range(q):
            pts_arr = line_points[(d, lv)]
            for i, pt in enumerate(pts_arr):
                other_pts[pt, d, :] = np.delete(pts_arr, i)

    # Vectorized line count computation
    def compute_line_counts(s_in):
        """Compute line_count[pt, d] = number of points in s_in on line (d) through pt, excluding pt."""
        line_count = np.zeros((total_points, n_dirs), dtype=np.int32)
        for d in range(n_dirs):
            for lv in range(q):
                pts_arr = line_points[(d, lv)]
                count_in_line = np.sum(s_in[pts_arr])
                # For each point on this line, the other points on the line are count_in_line - 1 (if pt is in s_in)
                # or count_in_line (if pt is not in s_in)
                # We want: for point pt on line (d,lv), line_count[pt, d] = (number of other points on line in s_in)
                # = count_in_line - s_in[pt]
                line_count[pts_arr, d] = count_in_line - s_in[pts_arr].astype(np.int32)
        return line_count

    def is_valid_add(s_in, line_count, free_count, pt):
        """Check if adding pt to s_in keeps the Nikodym property."""
        if free_count[pt] < 1:
            return False
        # Adding pt will decrement line_count for all other points on lines through pt
        # A point y becomes invalid if line_count[y,d] was 0 and free_count[y] was 1
        # (meaning y had exactly one direction with 0 other points, and that direction is affected)
        for d in range(n_dirs):
            others = other_pts[pt, d, :]
            # For each other point y, if line_count[y,d] == 0, then after adding pt it becomes 1
            # y is invalid if it had free_count[y] == 1 (only this direction had 0)
            affected = others[line_count[others, d] == 0]
            for y in affected:
                if free_count[y] == 1:
                    return False
        return True

    def greedy_build_vectorized(S_init, seed):
        rng = random.Random(seed)
        s_in = np.zeros(total_points, dtype=bool)
        for pt in S_init:
            s_in[pt] = True

        line_count = compute_line_counts(s_in)
        free_count = np.sum(line_count == 0, axis=1)
        current_S_size = int(np.sum(s_in))
        
        # Build candidate list
        candidates = [pt for pt in range(total_points) if not s_in[pt] and free_count[pt] >= 1]

        iteration = 0
        while candidates and time.time() - start_time < timeout - 30:
            iteration += 1
            
            # Choose a candidate: use damage-based heuristic sometimes
            if rng.random() < 0.4 and len(candidates) > 10:
                # Sample and pick lowest damage
                sample_size = min(30, len(candidates))
                sample = rng.sample(candidates, sample_size)
                best_pt = sample[0]
                best_damage = float('inf')
                for pt in sample:
                    damage = 0
                    for d in range(n_dirs):
                        others = other_pts[pt, d, :]
                        affected = others[line_count[others, d] == 0]
                        for y in affected:
                            if free_count[y] == 1:
                                damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_pt = pt
                pt = best_pt
            else:
                pt = rng.choice(candidates)

            if free_count[pt] < 1:
                candidates.remove(pt)
                continue

            # Check validity
            valid = True
            for d in range(n_dirs):
                others = other_pts[pt, d, :]
                affected = others[line_count[others, d] == 0]
                for y in affected:
                    if free_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                candidates.remove(pt)
                continue

            # Add pt
            s_in[pt] = True
            current_S_size += 1

            # Update line counts vectorized for this pt
            for d in range(n_dirs):
                others = other_pts[pt, d, :]
                # For each other point y on line d through pt:
                # line_count[y, d] increments by 1
                # if it was 0, free_count[y] decrements by 1
                was_zero = line_count[others, d] == 0
                line_count[others, d] += 1
                free_count[others[was_zero]] -= 1

            candidates.remove(pt)

            # Periodically rebuild candidate list
            if iteration % 50 == 0:
                candidates = [c for c in candidates if not s_in[c] and free_count[c] >= 1]

            if time.time() - start_time > timeout - 30:
                break

        return s_in, current_S_size

    def local_search_vectorized(s_in, S_size, seed, max_iterations=2000):
        """Local search with swaps to minimize S_size."""
        rng = random.Random(seed)
        line_count = compute_line_counts(s_in)
        free_count = np.sum(line_count == 0, axis=1)

        S_pts = [pt for pt in range(total_points) if s_in[pt]]
        non_S_pts = [pt for pt in range(total_points) if not s_in[pt]]

        no_improve = 0
        max_no_improve = 100

        while time.time() - start_time < timeout - 20 and no_improve < max_no_improve:
            improved = False
            for _ in range(max_iterations):
                if time.time() - start_time > timeout - 20:
                    break
                if not S_pts or not non_S_pts:
                    break
                rm_idx = rng.randrange(len(S_pts))
                rm_pt = S_pts[rm_idx]
                add_idx = rng.randrange(len(non_S_pts))
                add_pt = non_S_pts[add_idx]

                # Try swap: remove rm_pt, add add_pt
                s_in[rm_pt] = False
                # Update line counts for removal
                for d in range(n_dirs):
                    others = other_pts[rm_pt, d, :]
                    line_count[others, d] -= 1
                    became_zero = line_count[others, d] == 0
                    free_count[others[became_zero]] += 1

                # Check if add_pt is valid
                valid = free_count[add_pt] >= 1
                if valid:
                    for d in range(n_dirs):
                        others = other_pts[add_pt, d, :]
                        affected = others[line_count[others, d] == 0]
                        for y in affected:
                            if free_count[y] == 1:
                                valid = False
                                break
                        if not valid:
                            break

                if valid:
                    s_in[add_pt] = True
                    for d in range(n_dirs):
                        others = other_pts[add_pt, d, :]
                        was_zero = line_count[others, d] == 0
                        line_count[others, d] += 1
                        free_count[others[was_zero]] -= 1
                    S_pts[rm_idx] = add_pt
                    non_S_pts[add_idx] = rm_pt
                    improved = True
                    no_improve = 0
                else:
                    # Revert
                    s_in[rm_pt] = True
                    for d in range(n_dirs):
                        others = other_pts[rm_pt, d, :]
                        was_zero = line_count[others, d] == 0
                        line_count[others, d] += 1
                        free_count[others[was_zero]] -= 1

            if not improved:
                no_improve += 1

        return s_in, S_size

    # Generate diverse initial sets using different constructions
    def gen_init_unital(seed):
        """Unital-based initial set."""
        S_init = []
        rng = random.Random(seed)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        trace_x = (2 * xa) % p
                        norm_y = (ya * ya - 3 * yb * yb) % p
                        in_unital = (trace_x + norm_y) % p == 0
                        if in_unital:
                            pt = ((xa * p + xb) * q) + (ya * p + yb)
                            S_init.append(pt)
        # Random subset
        k = len(S_init) // 2
        return rng.sample(S_init, min(k, len(S_init)))

    def gen_init_random(seed):
        """Random initial set."""
        rng = random.Random(seed)
        # Start with a small random set
        k = rng.randint(50, 200)
        return rng.sample(range(total_points), k)

    def gen_init_grid(seed):
        """Grid-based initial set."""
        rng = random.Random(seed)
        S_init = []
        step = rng.randint(2, 5)
        for i in range(0, total_points, step):
            S_init.append(i)
        return S_init

    # Multi-restart strategy
    best_S_size = total_points
    best_s_in = None

    generators = [gen_init_unital, gen_init_random, gen_init_grid]
    num_restarts = 8

    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 60:
            break

        seed = 42 + restart * 1000
        gen_func = generators[restart % len(generators)]
        S_init = gen_func(seed)

        s_in, S_size = greedy_build_vectorized(S_init, seed)

        # Local search
        s_in, S_size = local_search_vectorized(s_in, S_size, seed + 1, max_iterations=500)

        # Final add phase: try to add more points
        line_count = compute_line_counts(s_in)
        free_count = np.sum(line_count == 0, axis=1)
        S_size = int(np.sum(s_in))
        candidates = [pt for pt in range(total_points) if not s_in[pt] and free_count[pt] >= 1]
        rng = random.Random(seed + 2)
        while candidates and time.time() - start_time < timeout - 30:
            pt = rng.choice(candidates)
            valid = True
            for d in range(n_dirs):
                others = other_pts[pt, d, :]
                affected = others[line_count[others, d] == 0]
                for y in affected:
                    if free_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                s_in[pt] = True
                S_size += 1
                for d in range(n_dirs):
                    others = other_pts[pt, d, :]
                    was_zero = line_count[others, d] == 0
                    line_count[others, d] += 1
                    free_count[others[was_zero]] -= 1
                candidates.remove(pt)
            else:
                candidates.remove(pt)

        if S_size < best_S_size:
            best_S_size = S_size
            best_s_in = s_in.copy()

    # If no valid set found, fallback
    if best_s_in is None:
        # Return all points (trivially valid but worst score)
        best_s_in = np.ones(total_points, dtype=bool)
        best_S_size = total_points

    # Build output: complement of N
    points = []
    for pt in range(total_points):
        if not best_s_in[pt]:
            xa = pt // (q * p)
            xb = (pt // q) % p
            ya = (pt // p) % p
            yb = pt % p
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Should not happen, but ensure at least one point
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
