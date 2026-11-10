import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin (evaluator kills at 630s)

    k_const = 3  # alpha^2 = 3

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

    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)

    # Precompute: for each point and direction, the list of other 48 points on that line
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

    def compute_line_counts(S_set):
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        for s_idx in S_set:
            for d_idx in range(num_dirs):
                for n_idx in affected[s_idx, d_idx]:
                    line_count[n_idx, d_idx] += 1
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    def is_valid(line_count):
        return np.all(np.sum(line_count == 0, axis=1) > 0)

    # Generate multiple seed candidates
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

    # Try different y-blocking sets
    y_block_candidates = []

    # 1. Squares in y_a
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))
    y_block_candidates.append(y_sq)

    # 2. Just y_a = 0
    y_zero = set()
    for v in range(p):
        y_zero.add((0, v))
    y_block_candidates.append(y_zero)

    # 3. y_a in {0, 1}
    y_01 = set()
    for v in range(p):
        y_01.add((0, v))
        y_01.add((1, v))
    y_block_candidates.append(y_01)

    # 4. y_a in {0, 1, 2}
    y_012 = set()
    for v in range(p):
        y_012.add((0, v))
        y_012.add((1, v))
        y_012.add((2, v))
    y_block_candidates.append(y_012)

    # 5. y_a is a non-square
    y_nonsq = set()
    for v in range(p):
        for a in {3, 5, 6}:
            y_nonsq.add((a, v))
    y_block_candidates.append(y_nonsq)

    # 6. y_a in {0, 3}
    y_03 = set()
    for v in range(p):
        y_03.add((0, v))
        y_03.add((3, v))
    y_block_candidates.append(y_03)

    # 7. y_a in {0, 1, 4} (squares including 0)
    y_sq_full = set()
    for v in range(p):
        for a in {0, 1, 2, 4}:
            y_sq_full.add((a, v))
    y_block_candidates.append(y_sq_full)

    # Also try different unital coefficients
    def make_seed_unital_variant(y_block_set, coeff):
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

    # Generate seeds and find the best initial one
    seeds = []
    for y_block in y_block_candidates:
        S = make_seed_unital_yblock(y_block)
        seeds.append(S)

    # Also try different coefficients
    for coeff in [1, 2, 3, 4, 5, 6]:
        for y_block in [y_sq, y_zero, y_01]:
            S = make_seed_unital_variant(y_block, coeff)
            seeds.append(S)

    # Evaluate all seeds and pick the best valid one
    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 60:
            break
        lc, zc = compute_line_counts(S)
        if is_valid(lc):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        # Fallback: use the known working seed
        best_seed = make_seed_unital_yblock(y_sq)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)

    # Local search with add and swap moves
    np.random.seed(42)
    random.seed(42)
    candidates = [i for i in range(2401) if i not in S]

    def try_add(p_idx):
        """Try to add p_idx to S. Returns (success, changes)"""
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
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

    def apply_remove(p_idx):
        S.remove(p_idx)
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                line_count[n_idx, d_idx] -= 1
                if line_count[n_idx, d_idx] == 0:
                    zero_counts[n_idx] += 1

    def revert_remove(p_idx):
        S.add(p_idx)
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                line_count[n_idx, d_idx] += 1
                if line_count[n_idx, d_idx] == 1:
                    zero_counts[n_idx] -= 1

    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1
        random.shuffle(candidates)
        made_progress = False

        # Try adds
        for p_idx in candidates:
            if time.time() > deadline - 30:
                break
            if p_idx in S:
                continue
            success, changes = try_add(p_idx)
            if success:
                apply_add(p_idx, changes)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                made_progress = True

        if not made_progress:
            # Try swaps: remove one point from S, try to add two points
            if len(S) > 0 and iteration % 3 == 0:
                s_list = list(S)
                random.shuffle(s_list)
                for rem_idx in s_list[:15]:  # try up to 15 removals
                    if time.time() > deadline - 30:
                        break
                    # Remove rem_idx
                    apply_remove(rem_idx)

                    # Try to add multiple points
                    adds = 0
                    random.shuffle(candidates)
                    for add_idx in candidates:
                        if time.time() > deadline - 30:
                            break
                        if add_idx in S:
                            continue
                        success, changes = try_add(add_idx)
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
                        # Revert: put rem_idx back
                        revert_remove(rem_idx)

        # Update candidates
        candidates = [i for i in range(2401) if i not in S]
        if len(candidates) == 0:
            break

    S = best_S

    # Build output: points NOT in S (the Nikodym set N)
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
