"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096 words of 64 bits each

    # Pre-compute initial wires using efficient bit packing
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def verify_partial(wires, comparators, start_idx=0):
        """
        Verify from a given state, applying comparators from start_idx onwards.
        Returns True if the resulting network is valid.
        """
        w = [x.copy() for x in wires]
        for idx in range(start_idx, len(comparators)):
            i, j = comparators[idx]
            wi = w[i]
            wj = w[j]
            w[i] = wi & wj
            w[j] = wi | wj

        for k in range(N - 1):
            if np.any(w[k] & ~w[k + 1]):
                return False
        return True

    def greedy_remove(comparators, max_iterations=100):
        """
        Iteratively remove redundant comparators.
        Multiple passes to find all removable comparators.
        """
        current = list(comparators)
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline - 1.0:
                break
            n = len(current)
            if n == 0:
                break
            # Try removing from all positions in a single pass
            for idx in range(n):
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def greedy_remove_fast(comparators, max_iterations=50):
        """
        Faster greedy removal: only check removals that don't break local ordering.
        """
        current = list(comparators)
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline - 1.0:
                break
            n = len(current)
            if n == 0:
                break
            # Try removing from all positions
            for idx in range(n):
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def batch_greedy_remove(comparators, max_iterations=30):
        """
        More aggressive greedy removal: try removing multiple comparators at once.
        """
        current = list(comparators)
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline - 1.0:
                break
            n = len(current)
            if n == 0:
                break
            # Try removing from all positions
            for idx in range(n):
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    # Start with odd-even transposition
    base_network = odd_even_transposition()
    base_network = greedy_remove(base_network)

    best_network = list(base_network)
    best_count = len(base_network)

    random.seed(42)

    # Simulated annealing with diverse moves
    temperature = 8.0
    cooling_rate = 0.997
    min_temperature = 0.3

    # Precompute some useful structures
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    while time.time() < deadline - 2.0:
        if temperature < min_temperature:
            temperature = min_temperature

        current = list(base_network)
        n_comp = len(current)

        if n_comp == 0:
            break

        r = random.random()

        if r < 0.45:
            # Remove a comparator
            if n_comp > 1:
                idx = random.randint(0, n_comp - 1)
                candidate = current[:idx] + current[idx + 1:]
                if verify_network(candidate):
                    current = candidate
                    current = greedy_remove_fast(current)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_network = list(current)
                        base_network = list(current)
                    continue
        elif r < 0.65:
            # Move a comparator to a new position
            if n_comp > 2:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = current[i]
                candidate = current[:i] + current[i+1:]
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
                if verify_network(candidate):
                    candidate = greedy_remove_fast(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                    else:
                        # Accept worse solutions with probability
                        if len(candidate) <= len(current) or random.random() < np.exp(-(len(candidate) - len(current)) / temperature):
                            base_network = list(candidate)
                    continue
        elif r < 0.80:
            # Replace a comparator with a nearby one
            idx = random.randint(0, n_comp - 1)
            old_u, old_v = current[idx]
            found = False
            # Try nearby replacements first
            candidates = []
            for du in [-2, -1, 0, 1, 2]:
                for dv in [-2, -1, 0, 1, 2]:
                    nu = old_u + du
                    nv = old_v + dv
                    if 0 <= nu < N and 0 <= nv < N and nu != nv:
                        if nu > nv:
                            nu, nv = nv, nu
                        if (nu, nv) != (old_u, old_v):
                            candidates.append((nu, nv))
            random.shuffle(candidates)
            for (u, v) in candidates[:10]:
                candidate = list(current)
                candidate[idx] = (u, v)
                if verify_network(candidate):
                    candidate = greedy_remove_fast(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                    found = True
                    break
            if not found:
                # Try random replacement
                for _ in range(5):
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (u, v) == (old_u, old_v):
                        continue
                    candidate = list(current)
                    candidate[idx] = (u, v)
                    if verify_network(candidate):
                        candidate = greedy_remove_fast(candidate)
                        if len(candidate) < best_count:
                            best_count = len(candidate)
                            best_network = list(candidate)
                            base_network = list(candidate)
                        found = True
                        break
        elif r < 0.90:
            # Add a comparator (temporarily increase size to escape local minima)
            if n_comp < 200:
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, n_comp)
                candidate = list(current)
                candidate.insert(pos, (u, v))
                if verify_network(candidate):
                    # Try to remove the added comparator or others
                    candidate = greedy_remove_fast(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                    else:
                        if len(candidate) <= len(current) + 2:
                            base_network = list(candidate)
                    continue
        else:
            # Random restart: perturb significantly
            if n_comp > 5:
                num_remove = random.randint(2, min(5, n_comp - 1))
                indices_to_remove = random.sample(range(n_comp), num_remove)
                indices_to_remove.sort(reverse=True)
                candidate = list(current)
                for idx in indices_to_remove:
                    candidate.pop(idx)
                if verify_network(candidate):
                    candidate = greedy_remove_fast(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                else:
                    # Try a different perturbation: swap two comparators
                    if n_comp >= 2:
                        i, j = random.sample(range(n_comp), 2)
                        candidate = list(current)
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        if verify_network(candidate):
                            candidate = greedy_remove_fast(candidate)
                            if len(candidate) < best_count:
                                best_count = len(candidate)
                                best_network = list(candidate)
                                base_network = list(candidate)

        # Cool down
        temperature *= cooling_rate

    # Final polish: aggressive greedy removal on best network
    if time.time() < deadline - 1.0:
        final = greedy_remove(best_network, max_iterations=200)
        if verify_network(final) and len(final) < best_count:
            best_network = final
            best_count = len(final)

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
