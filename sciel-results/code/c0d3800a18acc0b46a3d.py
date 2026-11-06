import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N

def _make_initial_inputs():
    idx = np.arange(NUM_INPUTS, dtype=np.uint8)
    return np.stack([(idx >> w) & 1 for w in range(N)], axis=0)

def _verify_network(comparators, inputs):
    state = inputs.copy()
    for i, j in comparators:
        mi = state[i] & state[j]
        mj = state[i] | state[j]
        state[i] = mi
        state[j] = mj
    for k in range(N - 1):
        if np.any((state[k] == 1) & (state[k + 1] == 0)):
            return False
    return True

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_network = _odd_even_transposition()
    best_count = len(best_network)
    inputs = _make_initial_inputs()

    if not _verify_network(best_network, inputs):
        raise RuntimeError("Seed failed")

    # Greedy removal: repeatedly remove any removable comparator
    improved = True
    while improved and (time.time() < deadline - 15):
        improved = False
        n = len(best_network)
        for idx in range(n):
            if time.time() > deadline - 15:
                break
            candidate = best_network[:idx] + best_network[idx + 1:]
            if _verify_network(candidate, inputs):
                best_network = candidate
                best_count = len(best_network)
                improved = True
                break

    # Random mutations to escape local optima
    if time.time() < deadline - 5:
        import random
        rng = random.Random(42)
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            candidate = best_network.copy()
            # Remove 1-3
            for _ in range(rng.randint(1, 3)):
                if candidate:
                    candidate.pop(rng.randint(0, len(candidate) - 1))
            # Add 1-3
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                candidate.insert(rng.randint(0, len(candidate)), (i, j))
            if len(candidate) < best_count and _verify_network(candidate, inputs):
                best_network = candidate
                best_count = len(best_network)
                # Greedy remove after improvement
                if time.time() < deadline - 5:
                    imp = True
                    while imp and (time.time() < deadline - 5):
                        imp = False
                        m = len(best_network)
                        for idx in range(m):
                            if time.time() > deadline - 5:
                                break
                            cand2 = best_network[:idx] + best_network[idx + 1:]
                            if _verify_network(cand2, inputs):
                                best_network = cand2
                                best_count = len(best_network)
                                imp = True
                                break

    # Final greedy removal
    if time.time() < deadline - 2:
        improved = True
        while improved and (time.time() < deadline - 2):
            improved = False
            n = len(best_network)
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                candidate = best_network[:idx] + best_network[idx + 1:]
                if _verify_network(candidate, inputs):
                    best_network = candidate
                    best_count = len(best_network)
                    improved = True
                    break

    return best_network
