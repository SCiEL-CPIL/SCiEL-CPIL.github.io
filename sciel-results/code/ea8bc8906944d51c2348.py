"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Use matrix multiplication: cost = sum(a * (b[np.ix_(p, p)]))
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)


def _sa_local_search(a, b, p, time_budget, rng):
    """Simulated annealing with 2-swap moves."""
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute for fast delta calculation
    # For a swap of facilities i and j (positions in p), 
    # the change in cost depends on the interactions.
    # Let's use a simpler approach: just compute the full cost change for 2-swap.
    # Actually, for efficiency, let's precompute row sums or use a delta formula.
    
    # For 2-swap of positions i, j (swapping p[i] and p[j]):
    # The cost change is complex. Let's just recompute for now if n is small,
    # but n=150 so full recompute is 150^2 = 22500 ops, which is fast enough.
    # Let's estimate: 150^2 = 22500 multiply-adds. In numpy, this is very fast.
    # We can do thousands of iterations per second.

    start_time = time.time()
    t = 1000.0  # initial temperature
    t_min = 1.0
    cooling_rate = 0.9999
    max_iterations = 100000

    for _ in range(max_iterations):
        if time.time() - start_time > time_budget:
            break

        # Choose two random positions to swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue

        # Perform swap
        p[i], p[j] = p[j], p[i]
        new_cost = _compute_cost(a, b, p)
        delta = new_cost - current_cost

        # Accept if better or with probability exp(-delta/t)
        if delta < 0 or rng.random() < np.exp(-delta / t):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Revert swap
            p[i], p[j] = p[j], p[i]

        t *= cooling_rate
        if t < t_min:
            t = t_min

    return best_p, best_cost


def _local_search_2swap(a, b, p, max_iter=10000):
    """Hill climbing with 2-swap moves."""
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    improved = True
    iter_count = 0

    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                p[i], p[j] = p[j], p[i]
                new_cost = _compute_cost(a, b, p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = p[j], p[i]
        if not improved:
            break

    return p, current_cost


def solve(a, b):
    """Solve the QAP using simulated annealing."""
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # leave 5s margin before 150s kill

    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()

    # Convert to numpy for internal use
    p_np = np.array(p, dtype=np.int64)

    # Multiple restarts with simulated annealing
    best_p = p_np.copy()
    best_cost = _compute_cost(a, b, p_np)

    # Use a fixed seed for reproducibility within the run, but different seeds for restarts
    rng = np.random.default_rng(42)

    # Phase 1: Simulated Annealing
    sa_time_budget = time_limit * 0.7
    ls_time_budget = time_limit * 0.3

    sa_p, sa_cost = _sa_local_search(a, b, p_np, sa_time_budget, rng)
    if sa_cost < best_cost:
        best_cost = sa_cost
        best_p = sa_p.copy()

    # Phase 2: Local search polish
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        ls_p, ls_cost = _local_search_2swap(a, b, best_p, max_iter=5000)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_p = ls_p

    return best_p.tolist()
# EVOLVE-BLOCK-END
