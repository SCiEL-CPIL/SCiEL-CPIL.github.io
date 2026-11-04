import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the bitmask of t-subsets it covers
    # and the list of t-subset indices
    k_sub_t_idxs = []
    block_masks = []
    for k_sub in all_k_subs:
        t_idxs = []
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
            mask |= (1 << idx)
        k_sub_t_idxs.append(t_idxs)
        block_masks.append(mask)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed, use_constrained=False):
        """Greedy set cover with bitmask acceleration."""
        rng = random.Random(seed)
        # Initialize uncovered mask: all bits set to 1
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            if use_constrained:
                # Find uncovered t-subset with fewest candidates
                # Sample a few random uncovered t-subsets and pick the most constrained
                best_s_idx = -1
                min_candidates = float('inf')
                for _ in range(20):
                    while True:
                        s_idx = rng.randint(0, n_t - 1)
                        if (uncovered >> s_idx) & 1:
                            break
                    candidates = [b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks]
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_s_idx = s_idx
                    if min_candidates == 1:
                        break
                s_idx = best_s_idx
            elif seed == 0:
                lsb = uncovered & -uncovered
                s_idx = lsb.bit_length() - 1
            else:
                # Pick a random t-subset index, ensure it's uncovered
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if (uncovered >> s_idx) & 1:
                        break

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Calculate gain using bitmask
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass using NumPy counts."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_blocks = []
            # Vectorized: for each block, check if any of its t-subsets has count==1
            # Build a matrix: rows = blocks, cols = their t-subsets (padded)
            # Actually, iterate but with early exit
            for b_idx in blocks_idx:
                # Check if any t-subset in this block has cover_count == 1
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if np.any(counts_for_block == 1):
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(blocks_idx, max_iter=50):
        """
        Local search: Try to remove blocks or swap them.
        Phase 1: Try removing each block; if gaps appear, try single-block repair.
        Phase 2: Try swapping blocks for better coverage.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        n_blocks = len(blocks_idx)
        
        # Precompute block masks as Python ints for fast bitwise ops
        # already have block_masks list
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Rebuild cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            # Phase 1: Try removing blocks
            for i in range(len(blocks_idx)):
                if time.time() >= deadline - 3.0:
                    break
                b_idx = blocks_idx[i]
                
                # Temporarily remove this block
                cover_count[k_sub_t_idxs[b_idx]] -= 1
                
                # Find gaps: t-subsets now uncovered
                gap_mask = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gap_mask |= (1 << t_idx)
                
                if gap_mask == 0:
                    # Safe to remove
                    blocks_idx.pop(i)
                    improved = True
                    # Restore cover_count for next iteration (rebuild at top of loop)
                    break
                else:
                    # Try to find a single replacement block that covers all gaps
                    # and is not already in blocks_idx
                    used_set = set(blocks_idx)
                    best_replace = -1
                    best_score = -1
                    
                    # Candidate blocks: those covering at least one gap t-subset
                    candidates = set()
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 0:
                            candidates.update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(b_idx)
                    candidates -= used_set
                    
                    # Limit candidates
                    if len(candidates) > 200:
                        rng = random.Random(iter_count * 1000 + i)
                        candidates = set(rng.sample(list(candidates), 200))
                    
                    for cand_idx in candidates:
                        if (block_masks[cand_idx] & gap_mask) != gap_mask:
                            continue
                        # Score: how many additional t-subsets it covers
                        score = (block_masks[cand_idx] & ~gap_mask).bit_count()
                        if score > best_score:
                            best_score = score
                            best_replace = cand_idx
                    
                    if best_replace != -1:
                        blocks_idx[i] = best_replace
                        improved = True
                        # cover_count will be rebuilt at top of next iteration
                        break
                    else:
                        # Restore
                        cover_count[k_sub_t_idxs[b_idx]] += 1
            
        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Randomized Greedy Restarts with different heuristics
    seeds = list(range(1, 101))
    for seed in seeds:
        if time.time() >= deadline - 30.0:
            break
        use_constrained = (seed % 3 == 0)
        blocks_idx = greedy_cover(seed, use_constrained)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(best_blocks)

    # 4. Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
