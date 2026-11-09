"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(pt, order):
    """Compute makespan for a job permutation on a flow shop. O(n) per call."""
    n = len(order)
    if n == 0:
        return 0.0
    cum = pt.cumsum(axis=0)  # shape (n, m), cum[i, k] = sum of first i jobs on machine k
    # For machine k, the completion of job j (position) is max(prev machine completion,
    # cumulative on this machine up to j)
    # We compute machine completion times sequentially.
    m = pt.shape[1]
    # comp[i, k] = completion of job at position i on machine k
    comp = np.empty((n, m), dtype=np.float64)
    # First machine: cumulative sums of selected jobs
    comp[:, 0] = cum[order, 0]
    for k in range(1, m):
        # comp[i, k] = max(comp[i-1, k], comp[i, k-1]) + pt[order[i], k]
        prev = np.zeros(n, dtype=np.float64)
        for i in range(n):
            j = order[i]
            val = max(prev[i - 1] if i > 0 else 0.0, comp[i, k - 1]) + pt[j, k]
            comp[i, k] = val
            prev[i] = val
    return comp[-1, m - 1]


def _makespan_fast(pt, order):
    """Vectorized-ish fast makespan using recurrence. O(n*m) but tight loop."""
    n = len(order)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # Use a 1D array for previous machine completions
    prev = np.zeros(n, dtype=np.float64)
    for k in range(m):
        cur = np.empty(n, dtype=np.float64)
        running = 0.0
        for i in range(n):
            j = order[i]
            val = (prev[i] if i > 0 else 0.0)
            if val < running:
                val = running
            val += pt[j, k]
            cur[i] = val
            running = val
        prev = cur
    return prev[n - 1]


def _neh(pt):
    """Neh heuristic for flow shop."""
    n = pt.shape[0]
    # Total processing time per job
    totals = pt.sum(axis=1)
    # Build the sequence: first job is the one with max(total, min(total, max(total)))
    # Standard NEH: first is argmax of (max(total, min(total, max(total))))
    # Actually NEH first job: argmax of max(total_j, min(total_j, max(total_j)))
    # Simpler: first job = argmax of (total_j) but NEH uses a specific rule.
    # We'll use: first = argmax of (total_j)
    first = int(np.argmax(totals))
    order = [first]
    remaining = list(range(n))
    remaining.remove(first)
    while remaining:
        # For each candidate, find best insertion position
        best_pos = -1
        best_ms = np.inf
        best_job = -1
        for j in remaining:
            for pos in range(len(order) + 1):
                cand = order[:pos] + [j] + order[pos:]
                ms = _makespan_fast(pt, cand)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
                    best_job = j
        order = order[:best_pos] + [best_job] + order[best_pos:]
        remaining.remove(best_job)
    return np.array(order, dtype=np.int64)


def _lpt(pt):
    """Longest processing time (sum) first."""
    return np.argsort(-pt.sum(axis=1)).astype(np.int64)


def _spt(pt):
    """Shortest processing time (sum) first."""
    return np.argsort(pt.sum(axis=1)).astype(np.int64)


def _or_opt_move(pt, order, pos, length):
    """Return new order after moving block of `length` jobs starting at `pos` to a random other position."""
    n = len(order)
    if length <= 0 or pos < 0 or pos + length > n:
        return order
    block = order[pos:pos + length]
    rest = np.concatenate([order[:pos], order[pos + length:]])
    # Insert block at a random position in rest
    insert_pos = np.random.randint(0, len(rest) + 1)
    new_order = np.concatenate([rest[:insert_pos], block, rest[insert_pos:]])
    return new_order


def _swap_move(pt, order, i, j):
    """Return new order after swapping positions i and j."""
    new_order = order.copy()
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow shop."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))

    rng = np.random.default_rng(42)

    # Generate initial solutions from multiple heuristics
    candidates = []
    try:
        candidates.append(_neh(pt))
    except Exception:
        pass
    candidates.append(_lpt(pt))
    candidates.append(_spt(pt))
    # Add a few random starts
    for _ in range(3):
        candidates.append(rng.permutation(n).astype(np.int64))

    # Evaluate all candidates and pick the best
    best_order = None
    best_ms = np.inf
    for cand in candidates:
        ms = _makespan_fast(pt, cand)
        if ms < best_ms:
            best_ms = ms
            best_order = cand.copy()

    if best_order is None:
        return list(range(n))

    # Simulated annealing + Or-opt local search
    current_order = best_order
    current_ms = best_ms
    best_order = current_order.copy()
    best_ms = current_ms

    # SA parameters
    T = max(best_ms * 0.1, 1.0)
    T_min = max(best_ms * 0.001, 0.01)
    cooling = 0.995

    # Local search: Or-opt (move block of 1, 2, 3 jobs)
    block_sizes = [1, 2, 3]

    steps = 0
    while time.time() < deadline:
        # Generate a move
        if rng.random() < 0.7:
            # Or-opt: move a block
            length = block_sizes[rng.integers(0, len(block_sizes))]
            pos = rng.integers(0, max(1, n - length))
            new_order = _or_opt_move(pt, current_order, int(pos), int(length))
        else:
            # Swap two positions
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            new_order = _swap_move(pt, current_order, int(i), int(j))

        new_ms = _makespan_fast(pt, new_order)
        delta = new_ms - current_ms
        steps += 1

        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order.copy()

        T = max(T * cooling, T_min)

        # If T is too low, restart with a perturbation (multi-restart)
        if T <= T_min * 1.01:
            # Perturb the best solution
            current_order = best_order.copy()
            for _ in range(rng.integers(2, 5)):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    current_order = _swap_move(pt, current_order, int(i), int(j))
            current_ms = _makespan_fast(pt, current_order)
            T = max(best_ms * 0.1, 1.0)

    return list(best_order.astype(int))
# EVOLVE-BLOCK-END
