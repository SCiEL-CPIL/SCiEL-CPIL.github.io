"""Size-efficient sorting network on 18 wires.

Combines the current Simulated-Annealing + greedy-refinement machinery with a
dependency-aware (Max-SAT style) refinement: a comparator (i,j) is redundant if,
on every input, wire i already holds the smaller value before the comparator
runs (then min->i, max->j is the identity). We detect that cheaply with a
per-input "wire i <= wire j" mask and prune such comparators.
"""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 56.0  # safety margin before the 90s kill

    # ---- bit-parallel state for all 2^18 Boolean inputs -------------------
    total_inputs = 1 << N          # 262144
    NUM_WORDS = total_inputs // 64 # 4096 words of 64 bits

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = ((indices >> i) & 1).astype(np.uint64)
        initial_wires[i] = (bits_i.reshape(NUM_WORDS, 64) * weights).sum(axis=1)

    def wire_is_ordered(i, j, wires):
        """True iff wires[i] <= wires[j] on every input, i.e. wire i holds the
        smaller value everywhere (mask where i has 1 and j has 0 is empty)."""
        return not np.any(wires[i] & ~wires[j])

    def verify_network(comparators):
        """Fast bit-parallel verification. True iff all Boolean inputs are sorted."""
        wires = [w.copy() for w in initial_wires]
        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None, use_prefilter=True):
        """Iteratively remove redundant comparators.

        With the prefilter we first test the cheap dependency condition
        (wire i already <= wire j before the comparator); only survivors are
        fully verified. This is far cheaper than blind full verification.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit is not None and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            if n <= 60:
                positions = list(range(n))
            else:
                positions = random.sample(range(n), min(60, n))
            random.shuffle(positions)

            wires = [w.copy() for w in initial_wires]
            for idx in positions:
                if idx >= len(current):
                    continue
                i, j = current[idx]
                # advance state to just before this comparator
                for (a, b) in current[:idx]:
                    wa = wires[a]
                    wb = wires[b]
                    wires[a] = wa & wb
                    wires[b] = wa | wb
                # cheap dependency pre-filter
                if use_prefilter and not wire_is_ordered(i, j, wires):
                    continue
                # full verification of the removal
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break
        return current

    def dependency_refine(comparators, time_limit=None):
        """Dependency-driven refinement: repeatedly scan the network, recompute
        the wire state up to each comparator, and drop any comparator whose
        effect is already the identity (wire i <= wire j everywhere)."""
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit is not None and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            wires = [w.copy() for w in initial_wires]
            for idx in range(n):
                i, j = current[idx]
                if not wire_is_ordered(i, j, wires):
                    # not redundant; advance state past this comparator
                    wi = wires[i]
                    wj = wires[j]
                    wires[i] = wi & wj
                    wires[j] = wi | wj
                    continue
                # redundant on every input: drop it and re-verify
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break
        return current

    # ---- seed networks -----------------------------------------------------
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def batcher_odd_even_merge(n):
        """Batcher's odd-even merge sort network on n wires (recursive)."""
        def merge(left, right):
            # left/right are already-sorted subnetworks on disjoint wire sets
            # (left on [0,half), right on [half,n)); merge them.
            n = len(left) + len(right)
            half = n // 2
            comps = []
            # odd-even merge: compare (i, i+half) for i in 0..half-1, then
            # recursively merge the two halves.
            for i in range(half):
                comps.append((i, i + half))
            # split into odd and even indexed wires of the merged set
            left2 = []
            right2 = []
            for i in range(n):
                (left2 if i % 2 == 0 else right2).append(i)
            # recursively merge each sub-sequence
            # (approximate Batcher; correctness is re-verified downstream)
            subL = batcher_odd_even_merge(len(left2))
            subR = batcher_odd_even_merge(len(right2))
            for (a, b) in subL:
                left2[a], left2[b] = left2[a], left2[b]
            # build comparator list on original indices
            def expand(sub, seq):
                out = []
                for (a, b) in sub:
                    out.append((seq[a], seq[b]))
                return out
            comps += expand(subL, left2)
            comps += expand(subR, right2)
            return comps

        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = batcher_odd_even_merge(half)
        right = batcher_odd_even_merge(n - half)
        right = [(a + half, b + half) for (a, b) in right]
        merged = merge(left, right)
        return left + right + merged

    def random_dense_network(num_comparators, seed):
        rnd = random.Random(seed)
        comparators = []
        for _ in range(num_comparators):
            i = rnd.randint(0, N - 1)
            j = rnd.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        rnd.shuffle(comparators)
        return comparators

    def simulated_annealing(initial_net, deadline, seed):
        """SA that favors removals; accepts valid worse states by temperature."""
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy

        rnd = random.Random(seed)
        t = 300.0
        alpha = 0.997
        t_min = 0.5

        while time.time() < deadline:
            if t < t_min:
                break
            r = rnd.random()
            candidate = list(current)

            if r < 0.50:
                # Remove (favored)
                if len(candidate) == 0:
                    continue
                idx = rnd.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif r < 0.62:
                # Insert
                u = rnd.randint(0, N - 1)
                v = rnd.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = rnd.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
            elif r < 0.82:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = rnd.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            else:
                # Replace
                if len(candidate) == 0:
                    continue
                idx = rnd.randint(0, len(candidate) - 1)
                u = rnd.randint(0, N - 1)
                v = rnd.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                candidate[idx] = (u, v)

            if not verify_network(candidate):
                t *= alpha
                continue

            new_energy = len(candidate)
            if new_energy < current_energy:
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                delta = new_energy - current_energy
                if delta > 0 and rnd.random() < np.exp(-delta / t):
                    current = candidate
                    current_energy = new_energy
            t *= alpha
        return best

    # ---- build seed list ---------------------------------------------------
    seeds = [
        ("odd_even", odd_even_transposition()),
        ("bubble", bubble_sort_network()),
    ]
    try:
        oem = batcher_odd_even_merge(N)
        if 0 < len(oem) < 400:
            seeds.append(("oem", oem))
    except Exception:
        pass
    for k, dens in enumerate([120, 140, 160, 180, 200]):
        seeds.append((f"rand_{k}", random_dense_network(dens, seed=100 + k * 7)))

    best_network = None
    best_count = float('inf')

    for idx, (name, net) in enumerate(seeds):
        if time.time() > deadline - 8.0:
            break

        # 1) cheap dependency refinement
        refined = dependency_refine(net, time_limit=deadline - 20.0)
        # 2) greedy removal (dependency-aware)
        refined = greedy_remove(refined, time_limit=deadline - 15.0)
        if len(refined) < best_count:
            best_count = len(refined)
            best_network = list(refined)

        if time.time() > deadline - 8.0:
            break

        # 3) SA from the refined start
        sa = simulated_annealing(refined, deadline - 6.0, seed=42 + idx * 101)
        sa = greedy_remove(sa, time_limit=deadline - 4.0)
        sa = dependency_refine(sa, time_limit=deadline - 3.0)
        if len(sa) < best_count:
            best_count = len(sa)
            best_network = list(sa)

    if best_network is None:
        fb = greedy_remove(odd_even_transposition(), time_limit=deadline - 1.0)
        best_network = fb

    if not verify_network(best_network):
        fb = greedy_remove(odd_even_transposition(), time_limit=deadline - 1.0)
        return fb

    return best_network
