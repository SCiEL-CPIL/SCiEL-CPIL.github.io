"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from multiprocessing import Pool, cpu_count

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 80.0  # seconds, leave 10s safety margin for 90s wall-clock limit

# Pre-compute initial wire bit-vectors globally for fast access
_INITIAL_WIRES = None

def _get_initial_wires():
    global _INITIAL_WIRES
    if _INITIAL_WIRES is None:
        _INITIAL_WIRES = []
        for i in range(N):
            half = 1 << i
            block = '0' * half + '1' * half
            reps = 1 << (N - i - 1)
            s = block * reps
            v = int(s, 2)
            _INITIAL_WIRES.append(v)
    return _INITIAL_WIRES

# EVOLVE-BLOCK-START
def _verify_worker(network):
    """Worker function for parallel verification (bit-vector zero-one principle)."""
    if not network:
        return N <= 1

    wires = _get_initial_wires()[:]

    for i, j in network:
        wi = wires[i]
        wj = wires[j]
        mask = wi & ~wj
        wires[i] = wi & ~mask
        wires[j] = wj | mask

    for i in range(N - 1):
        if wires[i] & ~wires[i + 1]:
            return False

    return True

def _odd_even_network(n):
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses parallel bit-vector verification and batched local search with
    mixed move types (remove, swap, move, replace, double-remove).
    """
    start_time = time.time()

    # Baseline: odd-even transposition (always valid)
    current = _odd_even_network(N)

    if not _verify_worker(current):
        return current

    current_count = len(current)
    best_count = current_count
    best_network = current[:]

    # Determine number of workers
    n_workers = min(cpu_count(), 8)

    def batch_greedy_remove(net, pool):
        """Repeatedly try to remove one comparator, verifying all candidates in parallel."""
        while True:
            if time.time() - start_time > DEADLINE - 5:
                break
            if len(net) == 0:
                break

            candidates = [net[:idx] + net[idx + 1:] for idx in range(len(net))]
            if not candidates:
                break

            results = pool.map(_verify_worker, candidates)

            found = False
            for idx, result in enumerate(results):
                if result:
                    net = candidates[idx]
                    found = True
                    break

            if not found:
                break

        return net

    with Pool(n_workers) as pool:
        # Phase 1: aggressive greedy cleanup from the seed
        current = batch_greedy_remove(current, pool)
        current_count = len(current)
        if current_count < best_count:
            best_count = current_count
            best_network = current[:]

        # Phase 2: randomized local search with mixed moves
        while time.time() - start_time < DEADLINE - 5:
            if len(current) == 0:
                break

            # Build a batch of candidate moves
            batch = []

            # Strategy A: single removals (prioritized)
            n_remove = max(1, len(batch) * 0) or 3
            for _ in range(3):
                if len(current) == 0:
                    break
                idx = random.randint(0, len(current) - 1)
                batch.append(current[:idx] + current[idx + 1:])

            # Strategy B: swaps (reorder)
            if len(current) >= 2:
                for _ in range(2):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        cand = current[:]
                        cand[i], cand[j] = cand[j], cand[i]
                        batch.append(cand)

            # Strategy C: moves (relocate)
            if len(current) > 1:
                for _ in range(2):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        cand = current[:]
                        comp = cand.pop(i)
                        cand.insert(j, comp)
                        batch.append(cand)

            # Strategy D: replace one comparator with a random one
            if len(current) > 0:
                for _ in range(2):
                    idx = random.randint(0, len(current) - 1)
                    new_i = random.randint(0, N - 2)
                    new_j = random.randint(new_i + 1, N - 1)
                    cand = current[:]
                    cand[idx] = (new_i, new_j)
                    batch.append(cand)

            # Strategy E: double removal (aggressive)
            if len(current) >= 2:
                for _ in range(2):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 2)
                    if j >= i:
                        j += 1
                    cand = current[:i] + current[i + 1:j] + current[j + 1:]
                    batch.append(cand)

            if not batch:
                break

            # Verify all candidates in parallel
            results = pool.map(_verify_worker, batch)

            # Accept the best (shortest) valid candidate in this batch
            best_idx = -1
            best_len = len(current)
            for idx, result in enumerate(results):
                if result and len(batch[idx]) < best_len:
                    best_len = len(batch[idx])
                    best_idx = idx

            if best_idx >= 0:
                current = batch[best_idx]
                current_count = len(current)
                if current_count < best_count:
                    best_count = current_count
                    best_network = current[:]
            else:
                # No move worked: perturb to escape local minimum
                # Swap 3-5 random pairs
                for _ in range(random.randint(3, 5)):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        current[i], current[j] = current[j], current[i]

                # Try to greedily remove from the perturbed state
                perturbed = batch_greedy_remove(current, pool)
                if len(perturbed) < current_count:
                    current = perturbed
                    current_count = len(current)
                    if current_count < best_count:
                        best_count = current_count
                        best_network = current[:]

    # Return the best network found
    return best_network


def verify_network_fast(network):
    """
    Verify network using bit-vector simulation with Python integers.
    """
    return _verify_worker(network)


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
