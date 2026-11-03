import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    block_weights = np.sum(triple_weights[block_triples], axis=1)

    block_triple_sets = [set(block_triples[i].tolist()) for i in range(n_blocks)]

    def greedy_cover(uncovered_init, mode="standard", random_tie=False, tie_temp=0.0, balance_alpha=0.0):
        """
        mode: 'standard', 'weighted', 'entropy', 'soft'
        balance_alpha: weight for entropy-balance penalty
        """
        uncovered = uncovered_init.copy()
        selected = []
        # Track how many times each triple has been covered (for entropy mode)
        cover_count = np.zeros(n_triples, dtype=np.int32)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if mode == "standard":
                gains = np.sum(uncovered[block_triples], axis=1)
            elif mode == "weighted":
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            elif mode == "entropy":
                # Base gain: cover uncovered triples, weighted by rarity
                base_gain = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
                # Penalty: prefer blocks that cover less-frequently-covered triples
                # Use current cover_count to compute a penalty
                # For each block, penalty = sum of (cover_count[triple] + 1) for triples in block
                # We want to minimize this, so subtract from gain
                penalty = np.sum(cover_count[block_triples].astype(np.float64) + 1.0, axis=1)
                gains = base_gain - balance_alpha * penalty / n_triples
            elif mode == "soft":
                # Soft surrogate: use a smoothed version of the max
                # Instead of hard argmax, use a temperature-based selection
                # Gain is weighted, but we add noise proportional to temperature
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tie:
                if tie_temp > 0 and mode != "soft":
                    # Softmax selection
                    exp_gains = np.exp(gains / tie_temp - max_gain / tie_temp)
                    total = np.sum(exp_gains)
                    if total < 1e-300:
                        break
                    probs = exp_gains / total
                    best_idx = np.random.choice(n_blocks, p=probs)
                elif mode == "soft" and tie_temp > 0:
                    # Add Gaussian noise to gains for soft selection
                    noisy_gains = gains + np.random.randn(n_blocks) * tie_temp
                    best_idx = np.argmax(noisy_gains)
                else:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            cover_count[block_triples[best_idx]] += 1

        return selected

    def prune_blocks(selected_indices):
        if not selected_indices:
            return selected_indices

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in selected_indices:
            cover_count[block_triples[b]] += 1

        remaining = list(selected_indices)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            for b in remaining:
                if time.time() > deadline:
                    break
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    changed = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

        return remaining

    def local_search_swap(selected_indices):
        remaining = list(selected_indices)
        n = len(remaining)
        selected_set = set(remaining)

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        def try_remove(pos):
            nonlocal n
            b = remaining[pos]
            for tj in block_triples[b]:
                if cover_count[tj] <= 1:
                    return False
            for tj in block_triples[b]:
                cover_count[tj] -= 1
            remaining.pop(pos)
            selected_set.discard(b)
            n -= 1
            return True

        def try_1for1(pos):
            nonlocal n
            b = remaining[pos]
            critical = []
            for tj in block_triples[b]:
                if cover_count[tj] == 1:
                    critical.append(tj)
            if not critical:
                return try_remove(pos)

            crit_set = set(critical)
            n_unselected = n_blocks - n
            if n_unselected <= 0:
                return False

            for _ in range(min(300, n_unselected)):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set:
                    continue
                if crit_set.issubset(block_triple_sets[nb]):
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    for tj in block_triples[nb]:
                        cover_count[tj] += 1
                    selected_set.discard(b)
                    selected_set.add(nb)
                    remaining[pos] = nb
                    return True
            return False

        def try_2for1(pos1, pos2):
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            unique_triples = []
            seen = set()
            for tj in block_triples[b1]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)
            for tj in block_triples[b2]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)

            if not unique_triples:
                for tj in block_triples[b1]:
                    cover_count[tj] -= 1
                for tj in block_triples[b2]:
                    cover_count[tj] -= 1
                hi = max(pos1, pos2)
                lo = min(pos1, pos2)
                remaining.pop(hi)
                remaining.pop(lo)
                selected_set.discard(b1)
                selected_set.discard(b2)
                n -= 2
                return True

            if len(unique_triples) > c_k_t:
                return False

            ut_set = set(unique_triples)
            n_unselected = n_blocks - n
            if n_unselected <= 0:
                return False

            for _ in range(min(400, n_unselected)):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set:
                    continue
                if ut_set.issubset(block_triple_sets[nb]):
                    for tj in block_triples[b1]:
                        cover_count[tj] -= 1
                    for tj in block_triples[b2]:
                        cover_count[tj] -= 1
                    for tj in block_triples[nb]:
                        cover_count[tj] += 1
                    selected_set.discard(b1)
                    selected_set.discard(b2)
                    selected_set.add(nb)
                    hi = max(pos1, pos2)
                    lo = min(pos1, pos2)
                    remaining.pop(hi)
                    remaining.pop(lo)
                    remaining.append(nb)
                    n -= 1
                    return True
            return False

        while time.time() < deadline - 2.0:
            improved = False

            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                if try_remove(i):
                    improved = True
                    break

            if improved:
                continue

            for _ in range(min(50, n)):
                if time.time() > deadline - 2.0:
                    break
                pos = random.randrange(n)
                if try_1for1(pos):
                    improved = True
                    break

            if improved:
                continue

            n_pairs_to_try = min(300, n * (n - 1) // 2)
            found_swap = False
            for _ in range(n_pairs_to_try):
                if time.time() > deadline - 2.0:
                    break
                i, j = random.sample(range(n), 2)
                if try_2for1(i, j):
                    found_swap = True
                    break

            if found_swap:
                continue

            extra_tries = min(150, n * (n - 1) // 2)
            for _ in range(extra_tries):
                if time.time() > deadline - 2.0:
                    break
                i, j = random.sample(range(n), 2)
                if try_2for1(i, j):
                    improved = True
                    break

            if not improved:
                break

        return remaining

    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Strategy 1: Standard greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), mode="standard", random_tie=False)
    try_solution(selected)

    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), mode="weighted", random_tie=False)
    try_solution(selected)

    # Strategy 3: Entropy-balanced greedy
    for alpha in [0.1, 0.3, 0.5]:
        if time.time() > deadline - 30.0:
            break
        selected = greedy_cover(np.ones(n_triples, dtype=bool), mode="entropy", random_tie=False, balance_alpha=alpha)
        try_solution(selected)

    # Strategy 4: Soft greedy with noise
    for temp in [0.5, 1.0, 2.0]:
        if time.time() > deadline - 25.0:
            break
        selected = greedy_cover(np.ones(n_triples, dtype=bool), mode="soft", random_tie=True, tie_temp=temp)
        try_solution(selected)

    # Strategy 5-14: Random tie-breaking restarts with varied modes
    modes = ["standard", "weighted", "entropy", "soft"]
    for restart in range(12):
        if time.time() > deadline - 20.0:
            break
        mode = modes[restart % 4]
        tie_temp = 0.0 if restart < 6 else (0.5 + restart * 0.3)
        balance_alpha = 0.3 if mode == "entropy" else 0.0
        selected = greedy_cover(np.ones(n_triples, dtype=bool), mode=mode,
                                random_tie=True, tie_temp=tie_temp, balance_alpha=balance_alpha)
        try_solution(selected)

    # Local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        improved = local_search_swap(best_indices)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]

    # Additional greedy restarts with local search if time remains
    if time.time() < deadline - 20.0:
        for extra_restart in range(8):
            if time.time() > deadline - 10.0:
                break
            mode = modes[extra_restart % 4]
            tie_temp = 0.5 + extra_restart * 0.3
            balance_alpha = 0.3 if mode == "entropy" else 0.0
            selected = greedy_cover(np.ones(n_triples, dtype=bool), mode=mode,
                                    random_tie=True, tie_temp=tie_temp, balance_alpha=balance_alpha)
            try_solution(selected)

        # Final local search on best
        if best_solution is not None and time.time() < deadline - 3.0:
            block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_to_idx[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        return [list(b) for b in blocks[:100]]

    return best_solution
