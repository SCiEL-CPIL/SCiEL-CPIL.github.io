import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, 10s buffer)

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for each possible block (7-subset) which t-subsets it covers
    # We'll work with block indices into the full list of C(24,7) blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build mapping: block_idx -> list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)

    # We'll track coverage counts for each t-subset
    # And maintain a set of selected blocks

    def try_randomized_greedy(seed, max_blocks=150):
        """Randomized greedy with rarity bias: prefer blocks covering rare triples."""
        rng = random.Random(seed)
        cover_counts = np.zeros(n_t, dtype=np.int32)
        selected = []
        n_covered = 0

        # Precompute: for each block, list of t-indices (already done)
        # We'll maintain a list of "active" blocks (not yet selected)
        # But that's 346k blocks, too many to shuffle each iteration.
        # Instead: at each step, sample a small random subset of candidate blocks
        # and pick the best one by rarity-weighted gain.

        for step in range(max_blocks):
            if time.time() > deadline - 10.0:
                break
            if n_covered == n_t:
                break

            # Sample a random pool of candidate blocks
            pool_size = min(5000, n_blocks)
            candidates = rng.sample(range(n_blocks), pool_size)

            best_block = -1
            best_score = -1

            for b_idx in candidates:
                t_idxs = block_t_indices[b_idx]
                # Compute gain: number of newly covered triples
                new_count = 0
                rarity_score = 0.0
                for ti in t_idxs:
                    if cover_counts[ti] == 0:
                        new_count += 1
                    # Rarity: lower count means more valuable
                    rarity_score += 1.0 / (cover_counts[ti] + 1)

                # Score: prioritize new coverage, then rarity
                # If a block covers any new triple, it's viable
                if new_count == 0:
                    continue
                score = new_count * 100.0 + rarity_score
                if score > best_score:
                    best_score = score
                    best_block = b_idx

            if best_block == -1:
                # Fallback: pick the block with max new coverage from full scan
                # (shouldn't happen if covering exists)
                best_block = -1
                best_new = 0
                for b_idx in range(n_blocks):
                    t_idxs = block_t_indices[b_idx]
                    new_count = sum(1 for ti in t_idxs if cover_counts[ti] == 0)
                    if new_count > best_new:
                        best_new = new_count
                        best_block = b_idx
                if best_block == -1:
                    break

            # Add the block
            selected.append(best_block)
            for ti in block_t_indices[best_block]:
                if cover_counts[ti] == 0:
                    n_covered += 1
                cover_counts[ti] += 1

        # Check validity
        if n_covered < n_t:
            return None

        return selected

    def reduce_solution(selected):
        """Remove redundant blocks via iterative deletion."""
        if selected is None:
            return None

        # Build cover counts
        cover_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for ti in block_t_indices[b_idx]:
                cover_counts[ti] += 1

        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try removing in reverse order
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                t_idxs = block_t_indices[b_idx]
                # Safe to remove if all its t-subsets are covered by others
                if np.min(cover_counts[t_idxs]) > 1:
                    cover_counts[t_idxs] -= 1
                    selected.pop(i)
                    improved = True
                    break

        # Verify
        if np.any(cover_counts == 0):
            return None
        return selected

    def local_swap_search(selected, max_iters=500):
        """Try swapping one block for another to enable further reductions."""
        if selected is None:
            return None
        rng = random.Random(42)

        cover_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for ti in block_t_indices[b_idx]:
                cover_counts[ti] += 1

        current_size = len(selected)
        best_size = current_size
        best_solution = selected.copy()

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if len(selected) >= best_size:
                continue

            # Pick a random block to try removing
            i = rng.randint(0, len(selected) - 1)
            b_idx = selected[i]
            t_idxs = block_t_indices[b_idx]

            # Find triples that would become uncovered
            essential = [ti for ti in t_idxs if cover_counts[ti] == 1]
            if not essential:
                # Can just remove
                cover_counts[t_idxs] -= 1
                selected.pop(i)
                continue

            # Find a replacement block that covers all essential triples
            # and doesn't add too much redundancy
            best_repl = -1
            best_repl_score = float('inf')

            # Sample candidates
            pool_size = min(2000, n_blocks)
            candidates = rng.sample(range(n_blocks), pool_size)
            for cand in candidates:
                if cand in selected:
                    continue
                cand_t_idxs = block_t_indices[cand]
                # Check if cand covers all essential
                cand_set = set(cand_t_idxs)
                if not all(e in cand_set for e in essential):
                    continue
                # Score: number of new redundancies introduced
                overlap = sum(1 for ti in cand_t_idxs if cover_counts[ti] > 1)
                if overlap < best_repl_score:
                    best_repl_score = overlap
                    best_repl = cand

            if best_repl != -1:
                # Perform swap
                for ti in t_idxs:
                    cover_counts[ti] -= 1
                selected[i] = best_repl
                for ti in block_t_indices[best_repl]:
                    cover_counts[ti] += 1

                # Now try to reduce again
                improved = True
                while improved:
                    improved = False
                    for j in range(len(selected) - 1, -1, -1):
                        bj = selected[j]
                        tj = block_t_indices[bj]
                        if np.min(cover_counts[tj]) > 1:
                            cover_counts[tj] -= 1
                            selected.pop(j)
                            improved = True
                            break

                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected.copy()

        return best_solution

    # Run multiple randomized trials with different seeds
    best_solution = None
    best_size = float('inf')

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break

        sol = try_randomized_greedy(seed, max_blocks=120)
        if sol is not None:
            reduced = reduce_solution(sol)
            if reduced is not None:
                # Try local swap search
                swapped = local_swap_search(reduced, max_iters=300)
                if swapped is not None and len(swapped) < best_size:
                    best_size = len(swapped)
                    best_solution = swapped.copy()
                elif len(reduced) < best_size:
                    best_size = len(reduced)
                    best_solution = reduced.copy()

    # Fallback: standard greedy if nothing worked
    if best_solution is None:
        cover_counts = np.zeros(n_t, dtype=np.int32)
        selected = []
        n_covered = 0
        for _ in range(150):
            if time.time() > deadline - 5.0:
                break
            if n_covered == n_t:
                break
            best_block = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                t_idxs = block_t_indices[b_idx]
                gain = sum(1 for ti in t_idxs if cover_counts[ti] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_gain == 0:
                break
            selected.append(best_block)
            for ti in block_t_indices[best_block]:
                if cover_counts[ti] == 0:
                    n_covered += 1
                cover_counts[ti] += 1
        best_solution = selected
        best_size = len(selected)

    # Final reduction
    if best_solution is not None:
        reduced = reduce_solution(best_solution)
        if reduced is not None:
            best_solution = reduced

    # Convert to list of lists
    result = [list(all_blocks[i]) for i in best_solution]

    # Final verification
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in best_solution:
        for ti in block_t_indices[b_idx]:
            test_counts[ti] += 1

    if np.any(test_counts == 0):
        # Should not happen, but fallback
        return [[0, 1, 2, 3, 4, 5, 6]]

    return result
