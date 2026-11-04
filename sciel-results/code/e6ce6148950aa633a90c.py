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

    def local_search(blocks_idx, max_iter=30):
        """
        Local search: Try to swap blocks to improve coverage.
        For each block, try replacing it with a different block that covers 
        the same essential t-subsets and possibly more.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Rebuild cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            for i, b_idx in enumerate(blocks_idx):
                # Find essential t-subsets (covered only by this block)
                essential_mask = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential_mask |= (1 << t_idx)
                
                if essential_mask == 0:
                    continue  # Not essential, skip
                
                # Try to find a replacement block that covers all essential t-subsets
                # and has higher total gain
                best_replace = -1
                best_score = -1
                
                # Sample candidate blocks
                candidates = set()
                for t_idx in k_sub_t_idxs[b_idx]:
                    candidates.update(t_sub_to_k_idxs[t_idx])
                candidates.discard(b_idx)
                
                # Limit candidates to speed up
                if len(candidates) > 100:
                    candidates = set(random.Random(iter_count).sample(list(candidates), 100))
                
                for cand_idx in candidates:
                    # Check if cand covers all essential t-subsets
                    if (block_masks[cand_idx] & essential_mask) != essential_mask:
                        continue
                    
                    # Score: number of new t-subsets covered (not in essential, not already covered by other blocks)
                    # Simple score: total coverage
                    score = block_masks[cand_idx].bit_count()
                    
                    if score > best_score:
                        best_score = score
                        best_replace = cand_idx
                
                if best_replace != -1 and best_replace not in blocks_idx:
                    # Check if replacement is valid (doesn't create duplicates)
                    if best_replace not in blocks_idx:
                        blocks_idx[i] = best_replace
                        improved = True
                        break
        
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
    seeds = list(range(1, 201))
    for seed in seeds:
        if time.time() >= deadline - 15.0:
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
