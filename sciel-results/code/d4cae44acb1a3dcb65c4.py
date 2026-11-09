import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a MODEL-AND-OPTIMIZE approach: parameterize the set N and optimize
    the complement size directly via greedy removal and local search.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s
    safety_margin = 5.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Element representation: (a, b) represents a + b*alpha
    # idx = a * p + b

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

    # Build multiple baseline Nikodym sets and keep the best after pruning
    
    def build_baseline(variant=0):
        """Build a baseline Nikodym set variant."""
        in_N_local = np.ones(total_points, dtype=bool)
        
        if variant == 0:
            # Original: Trace(x) + Norm(y) = 0 with special exclusions
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
                        in_N_local[p_idx] = False
        elif variant == 1:
            # Norm(x) + Trace(y) = 0
            x_special = set()
            for u in range(p):
                for v in range(p):
                    x_special.add(((u * u) % p, v))
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                norm_x = (xa * xa - xb * xb * alpha_sq) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    trace_y = (2 * ya) % p
                    p_idx = x_idx * q + y_idx
                    if (norm_x + trace_y) % p == 0 and (xa, xb) in x_special:
                        in_N_local[p_idx] = False
        elif variant == 2:
            # Trace(x) + Trace(y) = 0
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                trace_x = (2 * xa) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    trace_y = (2 * ya) % p
                    p_idx = x_idx * q + y_idx
                    if (trace_x + trace_y) % p == 0:
                        in_N_local[p_idx] = False
        elif variant == 3:
            # Norm(x) + Norm(y) = 0
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                norm_x = (xa * xa - xb * xb * alpha_sq) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (norm_x + norm_y) % p == 0:
                        in_N_local[p_idx] = False
        elif variant == 4:
            # Trace(x) * Norm(y) = 0
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                trace_x = (2 * xa) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (trace_x * norm_y) % p == 0:
                        in_N_local[p_idx] = False
        elif variant == 5:
            # Norm(x) * Trace(y) = 0
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                norm_x = (xa * xa - xb * xb * alpha_sq) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    trace_y = (2 * ya) % p
                    p_idx = x_idx * q + y_idx
                    if (norm_x * trace_y) % p == 0:
                        in_N_local[p_idx] = False
        elif variant == 6:
            # Trace(x) + Norm(y) = c for c=0,1 (union)
            for c in range(2):
                for x_idx in range(q):
                    xa, xb = elems[x_idx]
                    trace_x = (2 * xa) % p
                    for y_idx in range(q):
                        ya, yb = elems[y_idx]
                        norm_y = (ya * ya - yb * yb * alpha_sq) % p
                        p_idx = x_idx * q + y_idx
                        if (trace_x + norm_y) % p == c:
                            in_N_local[p_idx] = True
            # Reset to only keep where condition holds for some c
            in_N_local = np.zeros(total_points, dtype=bool)
            for c in range(2):
                for x_idx in range(q):
                    xa, xb = elems[x_idx]
                    trace_x = (2 * xa) % p
                    for y_idx in range(q):
                        ya, yb = elems[y_idx]
                        norm_y = (ya * ya - yb * yb * alpha_sq) % p
                        p_idx = x_idx * q + y_idx
                        if (trace_x + norm_y) % p == c:
                            in_N_local[p_idx] = True
        elif variant == 7:
            # Full plane (trivially valid, large N)
            in_N_local = np.ones(total_points, dtype=bool)
        elif variant == 8:
            # Trace(x) + Norm(y) = 0 without special exclusions
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                trace_x = (2 * xa) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (trace_x + norm_y) % p == 0:
                        in_N_local[p_idx] = True
            in_N_local = np.zeros(total_points, dtype=bool)
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                trace_x = (2 * xa) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (trace_x + norm_y) % p == 0:
                        in_N_local[p_idx] = True
        elif variant == 9:
            # Norm(x) + Norm(y) = 1
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                norm_x = (xa * xa - xb * xb * alpha_sq) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (norm_x + norm_y) % p == 1:
                        in_N_local[p_idx] = True
            in_N_local = np.zeros(total_points, dtype=bool)
            for x_idx in range(q):
                xa, xb = elems[x_idx]
                norm_x = (xa * xa - xb * xb * alpha_sq) % p
                for y_idx in range(q):
                    ya, yb = elems[y_idx]
                    norm_y = (ya * ya - yb * yb * alpha_sq) % p
                    p_idx = x_idx * q + y_idx
                    if (norm_x + norm_y) % p == 1:
                        in_N_local[p_idx] = True
        
        return in_N_local

    # Precompute line points: for each point and direction, the other q-1 points on the line
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)

    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    def prune_set(in_N_input):
        """Prune a Nikodym set by removing all points with support_count=0."""
        work = in_N_input.copy()
        
        # Compute valid directions
        ird = np.zeros((total_points, num_dirs), dtype=bool)
        for pt in range(total_points):
            for d in range(num_dirs):
                if np.all(work[line_points[pt, d, :]]):
                    ird[pt, d] = True
        
        # Check validity: every point must have at least one valid direction
        valid_count = np.sum(ird, axis=1)
        if np.any(valid_count == 0):
            return None  # Invalid set
        
        # Compute support counts
        sc = np.zeros(total_points, dtype=np.int32)
        for x in range(total_points):
            for d in range(num_dirs):
                if ird[x, d]:
                    for y in line_points[x, d, :]:
                        sc[y] += 1
        
        # Greedy removal of points with sc=0
        heap = []
        for pt in range(total_points):
            if work[pt]:
                heapq.heappush(heap, (sc[pt], pt))
        
        while heap:
            if time.time() > deadline - safety_margin:
                break
            sc_val, pt = heapq.heappop(heap)
            if not work[pt]:
                continue
            if sc[pt] != sc_val:
                continue
            if sc[pt] > 0:
                break
            work[pt] = False
        
        return work

    # Try multiple baseline variants and keep the best pruned result
    best_complement = 0
    best_in_N = None

    for variant in range(10):
        if time.time() > deadline - safety_margin:
            break
        
        in_N_input = build_baseline(variant)
        
        # Quick validity check before expensive pruning
        # Every point must be on at least one line whose other points are in N
        valid = True
        for pt in range(total_points):
            found = False
            for d in range(num_dirs):
                if np.all(in_N_input[line_points[pt, d, :]]):
                    found = True
                    break
            if not found:
                valid = False
                break
        
        if not valid:
            continue
        
        pruned = prune_set(in_N_input)
        if pruned is not None:
            comp = np.sum(~pruned)
            if comp > best_complement:
                best_complement = comp
                best_in_N = pruned.copy()

    # Fallback: if no variant worked, use the full plane
    if best_in_N is None:
        best_in_N = np.ones(total_points, dtype=bool)

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
