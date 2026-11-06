"""Size-efficient sorting network search for N=18 wires."""
import time
import numpy as np

N = 18
ALL_INPUTS = 1 << N
DEADLINE = 80.0  # seconds, safety margin before 90s kill

# Precompute the 2^18 x 18 boolean matrix once (bits of each input index)
def _build_all_inputs():
    idx = np.arange(ALL_INPUTS, dtype=np.uint8)
    # matrix[b, w] = bit w of b
    powers = (1 << np.arange(N, dtype=np.uint64))
    m = ((idx[:, None].astype(np.uint64) & powers[None, :]) != 0).astype(np.uint8)
    return m

def _apply_network(net, matrix):
    """Simulate network on all-boolean-inputs matrix. Returns final matrix."""
    state = matrix
    for (i, j) in net:
        a = state[:, i]
        b = state[:, j]
        # min -> i, max -> j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def _is_valid(net, matrix):
    """Check if net sorts all boolean inputs: after network, each row must be 0..0,1..1."""
    state = _apply_network(net, matrix)
    # For each row, find the last 0 and first 1; valid iff no 1 appears before a 0
    # Equivalent: for every row, state is non-increasing? No: min goes to i (lower wire), max to j.
    # After full sort, wire 0 has the global min, wire N-1 the global max.
    # So sorted row should be: some 0s then some 1s (non-decreasing).
    # Check: for each row, all 1s come after all 0s. i.e., state[:, k] <= state[:, k+1] for all k.
    # Actually sorted ascending: row[k] <= row[k+1].
    diff = state[:, 1:].astype(np.int8) - state[:, :-1].astype(np.int8)
    return bool(np.all(diff >= 0))

def _build_seed_network():
    """Odd-even transposition network (known valid, 153 comparators)."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _random_network(n_comp, existing=None):
    """Generate a random network of n_comp comparators."""
    net = []
    for _ in range(n_comp):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        net.append((i, j))
    return net

def construct():
    start = time.time()
    matrix = _build_all_inputs()

    # Seed: odd-even transposition
    best_net = _build_seed_network()
    best_score = len(best_net)

    # Verify seed (should be valid)
    if not _is_valid(best_net, matrix):
        # Fallback: just return seed as-is (it's known valid)
        return best_net

    np.random.seed(42)

    # Local search: try to remove comparators, then move them, then add
    current = list(best_net)
    current_score = len(current)

    def try_improve(net, matrix, deadline):
        """Local search: attempt to reduce comparator count."""
        net = list(net)
        n = len(net)
        improved = True
        while improved and (time.time() - start) < deadline:
            improved = False
            # Try removing each comparator
            for idx in range(len(net)):
                if (time.time() - start) > deadline:
                    break
                candidate = net[:idx] + net[idx+1:]
                if _is_valid(candidate, matrix):
                    net = candidate
                    improved = True
                    break
            # Try moving a comparator to a new position
            if not improved and (time.time() - start) < deadline:
                for idx in range(len(net)):
                    if (time.time() - start) > deadline:
                        break
                    i, j = net[idx]
                    # Try a few random new positions
                    for _ in range(3):
                        ni = np.random.randint(0, N)
                        nj = np.random.randint(0, N)
                        if ni == nj:
                            continue
                        if ni > nj:
                            ni, nj = nj, ni
                        if (ni, nj) == (i, j):
                            continue
                        candidate = net[:idx] + [(ni, nj)] + net[idx+1:]
                        if _is_valid(candidate, matrix):
                            net = candidate
                            improved = True
                            break
                    if improved:
                        break
        return net

    # Phase 1: Remove comparators (most impactful)
    current = try_improve(current, matrix, DEADLINE)
    if len(current) < best_score:
        best_net = current
        best_score = len(current)

    # Phase 2: More aggressive local search with moves
    if (time.time() - start) < DEADLINE:
        for _ in range(50):
            if (time.time() - start) > DEADLINE:
                break
            # Random perturbation: move a random comparator
            idx = np.random.randint(0, len(current))
            i, j = current[idx]
            ni = np.random.randint(0, N)
            nj = np.random.randint(0, N)
            if ni == nj:
                continue
            if ni > nj:
                ni, nj = nj, ni
            candidate = current[:idx] + [(ni, nj)] + current[idx+1:]
            if _is_valid(candidate, matrix):
                # Accept, then try to remove
                candidate = try_improve(candidate, matrix, DEADLINE)
                if len(candidate) < best_score:
                    best_net = candidate
                    best_score = len(candidate)
                    current = candidate

    # Phase 3: Try adding a comparator then removing two (net -1)
    if (time.time() - start) < DEADLINE:
        for _ in range(200):
            if (time.time() - start) > DEADLINE:
                break
            # Add a random comparator
            ni = np.random.randint(0, N)
            nj = np.random.randint(0, N)
            if ni == nj:
                continue
            if ni > nj:
                ni, nj = nj, ni
            candidate = current + [(ni, nj)]
            # Try to remove two comparators from this
            for r1 in range(len(candidate)):
                if (time.time() - start) > DEADLINE:
                    break
                for r2 in range(len(candidate)):
                    if r2 == r1:
                        continue
                    if (time.time() - start) > DEADLINE:
                        break
                    lo, hi = min(r1, r2), max(r1, r2)
                    candidate2 = candidate[:lo] + candidate[lo+1:hi] + candidate[hi+1:]
                    if len(candidate2) < best_score and _is_valid(candidate2, matrix):
                        best_net = candidate2
                        best_score = len(candidate2)
                        current = candidate2
                        break
                if len(current) < best_score:
                    break
            if len(current) < best_score:
                break

    # Final safety: ensure best_net is valid
    if not _is_valid(best_net, matrix):
        return _build_seed_network()

    return best_net
