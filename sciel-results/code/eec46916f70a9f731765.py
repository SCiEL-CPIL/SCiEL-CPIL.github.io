import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # For each triple, precompute which blocks cover it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tj in block_triples[i]:
            triple_to_blocks[tj].append(i)

    # Triple coverage frequency
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)

    # Precompute block weights
    block_weights = np.sum(triple_weights[block_triples], axis=1)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, tie_temp=0.0):
        """Run greedy set cover, optionally weighted with optional random tie-breaking."""
        uncovered = uncovered_init.copy()
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tie:
                if tie_temp > 0:
                    # Soft-max sampling: pick with probability proportional to exp(gain/temp)
                    exp_gains = np.exp(gains / tie_temp - max_gain / tie_temp)
                    total = np.sum(exp_gains)
                    if total < 1e-300:
                        break
                    probs = exp_gains / total
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices

        selected_set = set(selected_indices)
        # Count how many selected blocks cover each triple
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
                # Check if all triples in block b have cover_count > 1
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove block b
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    changed = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

        return remaining

    def local_search_swap(selected_indices):
        """Max-SAT style local search: iteratively try removals and 2-for-1 swaps
        until no improvement is found or time runs out. Hard constraint: every
        triple covered. Objective: minimize number of selected blocks."""
        remaining = list(selected_indices)
        n = len(remaining)
        selected_set = set(remaining)

        # Build coverage count: how many selected blocks cover each triple
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        # For fast lookup: map triple index -> list of selected block positions
        # Rebuild only when needed (after modifications)
        def rebuild_triple_to_selected():
            """Build mapping from triple -> list of (position, block_id) in remaining."""
            t2s = [[] for _ in range(n_triples)]
            for pos, b in enumerate(remaining):
                for tj in block_triples[b]:
                    t2s[tj].append(pos)
            return t2s

        def try_remove(pos):
            """Try removing block at position pos. Returns True if successful."""
            nonlocal n
            b = remaining[pos]
            # Check: all triples in b must have cover_count > 1
            for tj in block_triples[b]:
                if cover_count[tj] <= 1:
                    return False
            # Remove
            for tj in block_triples[b]:
                cover_count[tj] -= 1
            remaining.pop(pos)
            selected_set.discard(b)
            n -= 1
            return True

        def try_2for1(pos1, pos2):
            """Try removing blocks at pos1, pos2 and adding one block.
            Returns True if successful."""
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            # Find triples that would become uncovered: those with cover_count == 1
            # and present in b1 or b2
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
                # Both removable without replacement
                for tj in block_triples[b1]:
                    cover_count[tj] -= 1
                for tj in block_triples[b2]:
                    cover_count[tj] -= 1
                # Remove higher index first
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

            # Find an unselected block covering all unique_triples
            ut_set = set(unique_triples)
            # Try unselected blocks: use random sampling for speed, plus deterministic scan
            # Sample up to 500 random unselected blocks
            tried = set()
            for _ in range(min(500, n_blocks - n)):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set or nb in tried:
                    continue
                tried.add(nb)
                nb_set = set(block_triples[nb].tolist())
                if not ut_set.issubset(nb_set):
                    continue
                # Do the swap
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

            # Also try a deterministic scan of a subset of blocks
            for nb in range(0, n_blocks, max(1, n_blocks // 200)):
                if time.time() > deadline - 2.0:
                    return False
                if nb in selected_set or nb in tried:
                    continue
                nb_set = set(block_triples[nb].tolist())
                if not ut_set.issubset(nb_set):
                    continue
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

        # Main loop: iterate until no improvement or deadline
        while time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Try single removals (fast pass)
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                if try_remove(i):
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Try 2-for-1 swaps
            # Sample pairs to keep it tractable
            n_pairs_to_try = min(200, n * (n - 1) // 2)
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

            # No improvement found; try a few more random pairs with different sampling
            # before giving up (stochastic escape)
            extra_tries = min(100, n * (n - 1) // 2)
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
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    try_solution(selected)

    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 3-14: Random tie-breaking restarts with varying temperatures
    for restart in range(12):
        if time.time() > deadline - 30.0:
            break

        use_weighted = (restart % 2 == 0)
        tie_temp = 0.0 if restart < 6 else (0.5 + restart * 0.3)
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted,
                                random_tie=True, tie_temp=tie_temp)
        try_solution(selected)

    # Local search on the best solution found so far
    if best_solution is not None and time.time() < deadline - 5.0:
        # Reconstruct indices from best_solution
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
            # Vary the greedy strategy
            if extra_restart % 3 == 0:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=True, tie_temp=0.0)
            elif extra_restart % 3 == 1:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=True, tie_temp=1.0)
            else:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=True, tie_temp=2.0)
            try_solution(selected)

        # Final local search on best
        if best_solution is not None and time.time() < deadline - 3.0:
            block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_to_idx[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in blocks[:100]]

    return best_solution
