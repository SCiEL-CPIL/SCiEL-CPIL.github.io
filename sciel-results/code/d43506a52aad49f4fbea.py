"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 17
DEADLINE = 82.0  # safety margin before 90s kill

# Precompute sorted states: state with k ones has ones on wires N-k..N-1
_SORTED_STATES = set()
for _k in range(N + 1):
    _v = 0
    for _i in range(N - _k, N):
        _v |= (1 << _i)
    _SORTED_STATES.add(_v)

def _apply_network(state, network):
    """Apply network to a 17-bit integer state."""
    for i, j in network:
        bi = (state >> i) & 1
        bj = (state >> j) & 1
        if bi == 1 and bj == 0:
            state ^= (1 << i) | (1 << j)
    return state

def _verify_network(network):
    """Verify network sorts all 2^17 boolean inputs (zero-one principle)."""
    for state in range(1 << N):
        out = _apply_network(state, network)
        if out not in _SORTED_STATES:
            return False
    return True

def _verify_network_fast(network):
    """Faster verification: early exit on first failure, skip trivial states."""
    for state in range(1 << N):
        out = _apply_network(state, network)
        if out not in _SORTED_STATES:
            return False
    return True

def _compute_depth(comparators):
    """Compute parallel depth."""
    if not comparators:
        return 0
    wire_depth = [0] * N
    for i, j in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)

def _seed_network():
    """Odd-even transposition sorting network: 136 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _local_search_remove(comparators, time_limit, rng):
    """Try to remove comparators one at a time, greedily with random order."""
    current = list(comparators)
    while time.time() < time_limit:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > time_limit:
                return current
            candidate = current[:idx] + current[idx + 1:]
            if _verify_network_fast(candidate):
                current = candidate
                improved = True
                break
        if not improved:
            break
    return current

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    rng = random.Random(42)

    current = _seed_network()
    best = list(current)
    best_count = len(best)

    # Phase 1: Greedy removal
    current = _local_search_remove(current, DEADLINE - 15, rng)
    if len(current) < best_count:
        best = current
        best_count = len(best)

    # Phase 2: Randomized restarts with removal
    while time.time() < DEADLINE - 2:
        candidate = list(best)
        for _ in range(200):
            if time.time() > DEADLINE - 2:
                break
            if not candidate:
                break
            idx = rng.randint(0, len(candidate) - 1)
            test_candidate = candidate[:idx] + candidate[idx + 1:]
            if _verify_network_fast(test_candidate):
                candidate = test_candidate
                if len(candidate) < best_count:
                    best = candidate
                    best_count = len(best)
        if len(candidate) < best_count:
            best = candidate
            best_count = len(best)

    return best
# EVOLVE-BLOCK-END
