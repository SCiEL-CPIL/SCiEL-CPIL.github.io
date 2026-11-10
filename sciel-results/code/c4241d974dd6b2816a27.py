import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 570.0

    k_const = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # Directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)

    # Precompute affected: for each point and direction, the 48 other points
    affected = np.zeros((2401, num_dirs, 48), dtype=np.int32)
    for p_idx in range(2401):
        p_pt = idx_to_pt(p_idx)
        px, py = p_pt[0], p_pt[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    affected[p_idx, d_idx, count] = n_idx
                    count += 1

    # Flatten for faster vectorized operations
    # For each point p, all neighbors across all directions: affected[p].flatten() -> 2350 entries
    # We also need to know which direction each neighbor belongs to
    # affected shape: (2401, 50, 48)
    # Let's keep it as is but use numpy slicing for speed

    def compute_line_counts(S_set):
        """Compute line_count[x, d] = number of points in S on line through x in dir d (excluding x)"""
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        for s_idx in S_set:
            for d_idx in range(num_dirs):
                for n_idx in affected[s_idx, d_idx]:
                    line_count[n_idx, d_idx] += 1
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    def check_valid(zero_counts):
        return np.all(zero_counts > 0)

    # Generate seed candidates
    def make_seed_unital_yblock(y_block_set, coeff=2):
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (coeff * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    y_block_candidates = []
    # Squares in y_a
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))
    y_block_candidates.append(y_sq)
    # y_a = 0
    y_zero = set()
    for v in range(p):
        y_zero.add((0, v))
    y_block_candidates.append(y_zero)
    # y_a in {0, 1}
    y_01 = set()
    for v in range(p):
        y_01.add((0, v))
        y_01.add((1, v))
    y_block_candidates.append(y_01)
    # y_a in {0, 1, 2}
    y_012 = set()
    for v in range(p):
        y_012.add((0, v))
        y_012.add((1, v))
        y_012.add((2, v))
    y_block_candidates.append(y_012)
    # Non-squares
    y_nonsq = set()
    for v in range(p):
        for a in {3, 5, 6}:
            y_nonsq.add((a, v))
    y_block_candidates.append(y_nonsq)
    # y_a in {0, 3}
    y_03 = set()
    for v in range(p):
        y_03.add((0, v))
        y_03.add((3, v))
    y_block_candidates.append(y_03)
    # All squares including 0
    y_sq_full = set()
    for v in range(p):
        for a in {0, 1, 2, 4}:
            y_sq_full.add((a, v))
    y_block_candidates.append(y_sq_full)

    # Generate seeds
    seeds = []
    for y_block in y_block_candidates:
        S = make_seed_unital_yblock(y_block)
        seeds.append(S)
    for coeff in [1, 2, 3, 4, 5, 6]:
        for y_block in [y_sq, y_zero, y_01]:
            S = make_seed_unital_yblock(y_block, coeff)
            seeds.append(S)

    # Evaluate seeds
    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 60:
            break
        lc, zc = compute_line_counts(S)
        if check_valid(zc):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        best_seed = make_seed_unital_yblock(y_sq)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)

    # Local search with vectorized try_add
    np.random.seed(42)
    candidates = [i for i in range(2401) if i not in S]

    def try_add_vectorized(p_idx):
        """Try to add p_idx to S. Returns (success, changes)"""
        # For each direction d, for each neighbor n of p in direction d:
        # if line_count[n, d] == 0, then adding p would make it 1
        # this is only valid if zero_counts[n] > 1 (n still has another direction with count 0)
        # If zero_counts[n] == 1, adding p would make zero_counts[n] == 0 -> invalid
        changes = []
        for d_idx in range(num_dirs):
            neighbors = affected[p_idx, d_idx]
            for n_idx in neighbors:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        return False, None
                    else:
                        changes.append((n_idx, d_idx))
        return True, changes

    def apply_add(p_idx, changes):
        S.add(p_idx)
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    def try_remove_and_add(rem_idx, add_idx):
        """Try to remove rem_idx and add add_idx. Returns success."""
        # First, simulate removal
        # For each d, for each n in affected[rem_idx][d]:
        #   line_count[n, d] decreases by 1
        #   if it was 1, becomes 0, zero_counts[n] increases by 1
        # Then simulate addition of add_idx
        # For each d, for each n in affected[add_idx][d]:
        #   line_count[n, d] increases by 1
        #   if it was 0, becomes 1, zero_counts[n] decreases by 1
        # Need to check all zero_counts > 0 after both changes

        # Combine effects
        # Use a dict or array to track net changes
        # Since we're doing this one at a time, let's just simulate
        # But this is expensive. Let's do a simpler approach:
        # Just try adds greedily, and if stuck, do random restarts

    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1
        np.random.shuffle(candidates)
        made_progress = False

        # Try adds
        for p_idx in candidates:
            if time.time() > deadline - 30:
                break
            if p_idx in S:
                continue
            success, changes = try_add_vectorized(p_idx)
            if success:
                apply_add(p_idx, changes)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                made_progress = True

        if not made_progress:
            # Try swaps: remove one, add one or two
            if len(S) > 0 and iteration % 3 == 0:
                s_list = list(S)
                np.random.shuffle(s_list)
                for rem_idx in s_list[:20]:
                    if time.time() > deadline - 30:
                        break
                    # Remove rem_idx
                    S.remove(rem_idx)
                    for d_idx in range(num_dirs):
                        for n_idx in affected[rem_idx, d_idx]:
                            line_count[n_idx, d_idx] -= 1
                            if line_count[n_idx, d_idx] == 0:
                                zero_counts[n_idx] += 1

                    # Try to add multiple points
                    adds = 0
                    np.random.shuffle(candidates)
                    for add_idx in candidates:
                        if time.time() > deadline - 30:
                            break
                        if add_idx in S:
                            continue
                        success, changes = try_add_vectorized(add_idx)
                        if success:
                            apply_add(add_idx, changes)
                            adds += 1
                            if adds >= 2:
                                break

                    if adds >= 2:
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                    else:
                        # Revert
                        S.add(rem_idx)
                        for d_idx in range(num_dirs):
                            for n_idx in affected[rem_idx, d_idx]:
                                line_count[n_idx, d_idx] += 1
                                if line_count[n_idx, d_idx] == 1:
                                    zero_counts[n_idx] -= 1

        # Update candidates
        candidates = [i for i in range(2401) if i not in S]
        if len(candidates) == 0:
            break

    S = best_S

    # Build output
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
