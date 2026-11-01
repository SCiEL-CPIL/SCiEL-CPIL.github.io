import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Build coverage matrix: n_ksubs x n_tsubs (uint8), where M[j, i] = 1 if k-sub j covers t-sub i
    #    This is memory-heavy but enables vectorized operations
    #    For v=19, k=7, t=4: n_ksubs = C(19,7) = 50388, n_tsubs = C(19,4) = 3876
    #    Matrix: 50388 * 3876 bytes = ~195 MB as uint8 - manageable
    cov_matrix = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for j, ks in enumerate(all_ksubs):
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            cov_matrix[j, idx] = 1

    # Precompute row sums (total coverage of each block)
    total_cov = cov_matrix.sum(axis=1)

    def safe_fallback():
        """
        Generates a valid covering with at most C(v,t) blocks.
        Strategy: For each t-subset, pick one k-subset containing it.
        Collect unique k-subsets.
        """
        blocks_set = set()
        for tsub in all_tsubs:
            tset = set(tsub)
            # Pick k-t elements from the remaining v-t elements
            # Deterministic choice to ensure reproducibility if needed, 
            # but random is fine too. Let's use sorted first available for simplicity.
            remaining_points = [p for p in range(v) if p not in tset]
            extra = remaining_points[:k-t]
            block = tuple(sorted(list(tset) + extra))
            blocks_set.add(block)
        return [list(b) for b in blocks_set]

    def is_valid_cover(blocks):
        """Check if the list of blocks covers all t-subsets."""
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                covered[tsub_to_idx[tsub]] = True
        return covered.all()

    def greedy_cover(seed, max_time):
        """Run greedy set cover with incremental gain updates and randomized tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = n_tsubs
        used = np.zeros(n_ksubs, dtype=bool)
        selected = []
        t_start = time.time()
        
        # Initialize gains
        gains = cov_matrix.sum(axis=1).astype(np.int32)
        
        while remaining > 0 and time.time() < max_time:
            # Find candidates with positive gain among unused blocks
            gains[used] = 0
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomized selection: prefer max gain, but with some randomness
            if len(candidates) > 1:
                # Weighted random: higher gain blocks more likely, but allow some diversity
                # Use a softmax-like selection among top candidates
                top_n = min(10, len(candidates))
                top_candidates = candidates[:top_n] if len(candidates) <= top_n else candidates[rng.choice(len(candidates), size=top_n, replace=False)]
                chosen_idx = top_candidates[rng.randint(len(top_candidates))]
            else:
                chosen_idx = candidates[0]

            selected.append(chosen_idx)
            used[chosen_idx] = True
            
            # Update gains incrementally: for each unused block, subtract overlap with newly selected block
            new_block_cov = cov_matrix[chosen_idx]
            # Only update gains for unused blocks
            unused_mask = ~used
            overlap = (cov_matrix[unused_mask] & new_block_cov[None, :]).sum(axis=1)
            gains_indices = np.where(unused_mask)[0]
            gains[gains_indices] -= overlap
            
            uncovered = uncovered & (1 - new_block_cov)
            remaining = uncovered.sum()

        return selected

    def remove_redundant(selected, max_time):
        """Remove redundant blocks from the selected set using coverage counts."""
        if not selected or len(selected) <= 1:
            return list(selected)

        active = list(selected)
        n_active = len(active)
        
        # Build coverage count array: for each t-sub, how many active blocks cover it
        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in active:
            coverage_count += cov_matrix[bi]
        
        t_start = time.time()
        changed = True

        while changed and time.time() < max_time:
            changed = False
            new_active = []
            
            for bi in active:
                # Check if bi is redundant: all t-subs it covers have coverage_count > 1
                bi_cov = cov_matrix[bi]
                # If any t-sub covered by bi has coverage_count == 1, bi is essential
                if (bi_cov & (coverage_count == 1)).sum() == 0:
                    # bi is redundant - remove it
                    changed = True
                    coverage_count -= bi_cov
                else:
                    new_active.append(bi)

            active = new_active

        return active

    # Main search: multiple restarts with different random seeds
    best_blocks = None
    best_count = float('inf')

    # First greedy run
    t0 = time.time()
    quick_selected = greedy_cover(0, min(30.0, deadline - 10.0))
    t1 = time.time()
    quick_time = t1 - t0

    if quick_selected:
        quick_active = remove_redundant(quick_selected, deadline - 5.0)
        if quick_active:
            candidate_blocks = [list(all_ksubs[idx]) for idx in quick_active]
            # Validate
            if is_valid_cover(candidate_blocks) and len(candidate_blocks) <= n_tsubs:
                best_blocks = candidate_blocks
                best_count = len(candidate_blocks)

    # Now do additional restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Estimate how many restarts we can afford
        estimated_per_restart = max(quick_time, 1.0)
        n_restarts = min(30, int(remaining_time / estimated_per_restart))
        
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            seed = r + 1
            # Give each restart a reasonable time budget
            time_budget = min(deadline - time.time() - 2.0, max(5.0, quick_time * 1.5))
            if time_budget < 2.0:
                break

            selected = greedy_cover(seed, time_budget)
            if not selected:
                continue

            active = remove_redundant(selected, deadline - 2.0)
            if active:
                candidate_blocks = [list(all_ksubs[idx]) for idx in active]
                if is_valid_cover(candidate_blocks) and len(candidate_blocks) <= n_tsubs and len(candidate_blocks) < best_count:
                    best_count = len(candidate_blocks)
                    best_blocks = candidate_blocks

    if best_blocks is None:
        # Fallback: return a valid covering within size limit
        best_blocks = safe_fallback()

    return best_blocks
