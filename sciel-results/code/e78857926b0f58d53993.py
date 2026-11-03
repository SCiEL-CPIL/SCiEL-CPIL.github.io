import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
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

    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    # Weight: inverse frequency (rarity)
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)

    def greedy_cover(use_weighted=False, random_tiebreak=False, seed=None):
        """Run greedy set cover with optional weighting and random tie-breaking."""
        if seed is not None:
            np.random.seed(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            # Calculate gains
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tiebreak:
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution using coverage counts."""
        if not block_indices:
            return block_indices

        n_sel = len(block_indices)
        # coverage_count[j] = how many selected blocks cover triple j
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1

        keep = []
        for i in range(n_sel):
            my_triples = block_triples[block_indices[i]]
            # Check if all triples in this block are covered by at least one OTHER block
            # i.e., coverage_count[triple] > 1 for all triples in this block
            if np.all(coverage_count[my_triples] > 1):
                # Safe to remove
                coverage_count[my_triples] -= 1
            else:
                keep.append(block_indices[i])

        return keep

    def local_search_swap(block_indices, max_iters=500):
        """Try to reduce block count by swapping one block out for another.
        Returns improved list of block indices."""
        if not block_indices:
            return block_indices

        sel = list(block_indices)
        n_sel = len(sel)

        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in sel:
            coverage_count[block_triples[bidx]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 10.0:
                break

            # Try removing each block and see if we can cover the newly uncovered triples
            for i in range(n_sel):
                if time.time() > deadline - 10.0:
                    break
                my_triples = block_triples[sel[i]]
                # Temporarily remove block i
                coverage_count[my_triples] -= 1

                # Find triples that are now uncovered
                uncovered_mask = coverage_count[my_triples] == 0
                if not np.any(uncovered_mask):
                    # Block i was fully redundant - remove it
                    sel.pop(i)
                    n_sel -= 1
                    improved = True
                    continue

                uncovered_triples = my_triples[uncovered_mask]
                # Find blocks that cover all uncovered_triples
                # For each candidate block, check if it covers all uncovered_triples
                # and doesn't add too much overhead
                best_candidate = -1
                best_cost = float('inf')

                # Quick filter: candidate must cover all uncovered_triples
                # Check overlap: for each uncovered triple, count how many candidates cover it
                # Use a faster approach: for each candidate, count how many uncovered_triples it covers
                for cand in range(n_blocks):
                    cand_triples = block_triples[cand]
                    # Check if cand covers all uncovered_triples
                    # uncovered_triples is a small array
                    if len(uncovered_triples) == 0:
                        continue
                    # Check coverage
                    covered = 0
                    for ut in uncovered_triples:
                        if ut in cand_triples:
                            covered += 1
                    if covered == len(uncovered_triples):
                        # Cost: number of NEW triples this block adds (not already covered)
                        new_triples = np.sum(coverage_count[cand_triples] == 0)
                        cost = new_triples
                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = cand

                if best_candidate >= 0:
                    # Replace sel[i] with best_candidate
                    sel[i] = best_candidate
                    coverage_count[block_triples[best_candidate]] += 1
                    improved = True
                else:
                    # Restore
                    coverage_count[my_triples] += 1

        return sel

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')

    def try_solution(block_indices):
        nonlocal best_solution, best_size
        if not block_indices:
            return
        peeled = peel_redundant(block_indices)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(blocks[i]) for i in peeled]

    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    try_solution(sel)

    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    try_solution(sel)

    # Strategy 3+: Random restarts with weighted greedy
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, seed=r * 1000 + 42)
        try_solution(sel)

    # Local search polish on the best solution
    if best_solution is not None:
        # Convert back to indices
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        current_indices = [block_to_idx[tuple(b)] for b in best_solution]

        # Multiple passes of local search
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            improved = local_search_swap(current_indices, max_iters=200)
            current_indices = improved
            # Peel again
            peeled = peel_redundant(current_indices)
            if len(peeled) < best_size:
                best_size = len(peeled)
                best_solution = [list(blocks[i]) for i in peeled]

    if best_solution is None:
        best_solution = []

    return best_solution
