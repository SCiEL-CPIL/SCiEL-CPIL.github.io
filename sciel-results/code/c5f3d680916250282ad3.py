import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a hypergraph vertex-cover formulation with efficient precomputation
    of co-occurrence matrices, followed by greedy pruning and local search.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    safety_margin = 10.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    elems = [idx_to_elem(i) for i in range(q)]

    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50 directions
    total_points = q * q  # 2401

    # Build baseline Nikodym set
    in_N = np.ones(total_points, dtype=bool)
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = elems[x_idx]
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = elems[y_idx]
            norm_y = (ya * ya - yb * yb * alpha_sq) % p
            p_idx = x_idx * q + y_idx
            if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                in_N[p_idx] = False

    # Precompute line points
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Initialize validity tracking (vectorized)
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for d in range(num_dirs):
        all_in = np.all(in_N[line_points[:, d, :]], axis=1)
        is_valid_dir[:, d] = all_in
        valid_count += all_in.astype(np.int32)

    # Precompute co-occurrence matrix: cooccur[y, x, d] = 1 if y is on line through x in direction d
    # This is key for efficient removability checking
    cooccur = np.zeros((total_points, total_points, num_dirs), dtype=np.int8)
    for y in range(total_points):
        for d in range(num_dirs):
            xs = line_points[y, d, :]
            cooccur[y, xs, d] = 1

    # Also precompute for each point y, which (x, d) pairs have y on their line
    # line_partners[y] = list of (x, d) where y is on line through x in direction d
    line_partners = [[] for _ in range(total_points)]
    for x in range(total_points):
        for d in range(num_dirs):
            for y in line_points[x, d, :]:
                line_partners[y].append((x, d))

    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def try_prune(order, max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        removed = 0

        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            progress = False
            for y in order:
                if not work[y]:
                    continue

                # Check removability using precomputed co-occurrence
                # y is removable if for all (x, d) where y is on line through x in dir d:
                #   if ird[x, d] is True, then x has another valid direction d2
                #   such that y is NOT on line through x in dir d2
                removable = True
                for x, d in line_partners[y]:
                    if ird[x, d]:
                        # Check if x has another valid direction not containing y
                        has_other = False
                        for d2 in range(num_dirs):
                            if d2 != d and ird[x, d2] and not cooccur[y, x, d2]:
                                has_other = True
                                break
                        if not has_other:
                            removable = False
                            break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    # Update validity: removing y invalidates (x, d) for all
                    # (x, d) where y is on line through x in dir d
                    for x, d in line_partners[y]:
                        if ird[x, d]:
                            ird[x, d] = False
                            wc[x] -= 1

            if not progress:
                break

        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            best_valid_count = wc.copy()
            best_is_valid_dir = ird.copy()

        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    def compute_coverage(work, ird, wc):
        """Compute coverage for each point: number of valid (x,d) pairs where x is on a line through y."""
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not work[y]:
                continue
            for x, d in line_partners[y]:
                if ird[x, d]:
                    coverage[y] += 1
        return coverage

    # Strategy 1: Sequential order
    order1 = list(range(total_points))
    try_prune(order1, max_rounds=3)

    # Strategy 2: Random order
    if time.time() < deadline - safety_margin:
        rng = np.random.RandomState(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=3)

    # Strategy 3: Prioritize points with low coverage
    if time.time() < deadline - safety_margin:
        coverage = compute_coverage(in_N, is_valid_dir, valid_count)
        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune(order3, max_rounds=2)

    # Strategy 4: More random restarts
    if time.time() < deadline - safety_margin:
        seeds = [123, 456, 789, 101, 202, 303, 404, 505, 606, 707]
        for seed in seeds:
            if time.time() > deadline - safety_margin:
                break
            rng = np.random.RandomState(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune(order4, max_rounds=2)

    # Strategy 5: Coverage-based ordering with multiple passes
    if time.time() < deadline - safety_margin:
        for pass_num in range(3):
            if time.time() > deadline - safety_margin:
                break
            coverage = compute_coverage(in_N, is_valid_dir, valid_count)
            active_points = [y for y in range(total_points) if in_N[y]]
            active_points.sort(key=lambda y: coverage[y])
            try_prune(active_points, max_rounds=2)

    # Strategy 6: Local search - try swapping points
    if time.time() < deadline - safety_margin:
        for seed in range(5):
            if time.time() > deadline - safety_margin:
                break
            rng = np.random.RandomState(2000 + seed)
            work = best_in_N.copy()
            wc = best_valid_count.copy()
            ird = best_is_valid_dir.copy()
            
            # Try to remove random points and see if we can add others back
            for _ in range(200):
                if time.time() > deadline - safety_margin:
                    break
                candidates = [y for y in range(total_points) if work[y]]
                if not candidates:
                    break
                y = candidates[rng.randint(len(candidates))]
                
                # Check if y is removable
                removable = True
                for x, d in line_partners[y]:
                    if ird[x, d]:
                        has_other = False
                        for d2 in range(num_dirs):
                            if d2 != d and ird[x, d2] and not cooccur[y, x, d2]:
                                has_other = True
                                break
                        if not has_other:
                            removable = False
                            break
                
                if removable:
                    work[y] = False
                    for x, d in line_partners[y]:
                        if ird[x, d]:
                            ird[x, d] = False
                            wc[x] -= 1

            comp = np.sum(~work)
            if comp > best_complement:
                best_complement = comp
                best_in_N = work.copy()
                best_valid_count = wc.copy()
                best_is_valid_dir = ird.copy()

    # Final validity check
    if np.any(valid_count == 0):
        in_N = np.ones(total_points, dtype=bool)
        for x_idx in range(q):
            xa, xb = elems[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = elems[y_idx]
                norm_y = (ya * ya - yb * yb * alpha_sq) % p
                p_idx = x_idx * q + y_idx
                if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                    in_N[p_idx] = False
        valid_count = np.zeros(total_points, dtype=np.int32)
        is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
        for d in range(num_dirs):
            all_in = np.all(in_N[line_points[:, d, :]], axis=1)
            is_valid_dir[:, d] = all_in
            valid_count += all_in.astype(np.int32)
        best_in_N = in_N.copy()
        best_valid_count = valid_count.copy()
        best_is_valid_dir = is_valid_dir.copy()
        best_complement = np.sum(~in_N)

    # Extract points
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
