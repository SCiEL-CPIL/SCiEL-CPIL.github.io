import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def greedy_cover(seed, randomness=0.0):
        """
        Greedy set cover with bitmask acceleration.
        randomness: 0.0 is pure greedy, 1.0 is random selection from candidates.
        """
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            s_idx = -1
            while s_idx == -1:
                s_idx = rng.randint(0, n_t - 1)
                if not ((uncovered >> s_idx) & 1):
                    s_idx = -1
                    continue
                break
            
            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1
            best_gain_second = -1 # For diversity
            
            # Calculate gains for all candidates covering this specific s_idx
            gains = []
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                gains.append((gain, b_idx))
            
            if not gains:
                break
                
            # Sort by gain
            gains.sort(reverse=True)
            
            if randomness == 0.0:
                best_gain, best_block_idx = gains[0]
            else:
                # Pick from top candidates with some probability
                if randomness > 0.5 and len(gains) > 1:
                    top_k = min(5, len(gains))
                    _, best_block_idx = rng.choice(gains[:top_k])
                else:
                    best_gain, best_block_idx = gains[0]

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
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

    def lp_relaxation_lower_bound(essential_blocks, uncovered_mask, time_limit=0.5):
        """
        Compute a lower bound on the number of additional blocks needed to cover
        all t-subsets in uncovered_mask, given that essential_blocks are already selected.
        Uses a simple LP relaxation via subgradient method.
        """
        if uncovered_mask == 0:
            return 0
        
        # Get list of uncovered t-subset indices
        uncovered_idxs = []
        temp = uncovered_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            uncovered_idxs.append(idx)
            temp ^= lsb
        
        if not uncovered_idxs:
            return 0
            
        # For each uncovered t-subset, count how many non-essential blocks can cover it
        # This gives a simple lower bound: sum over uncovered t-subsets of 1 / (number of covering blocks)
        # But we need to be more careful.
        
        # Simple approach: For each uncovered t-subset, count available blocks that cover it
        # and are not in essential_blocks
        essential_set = set(essential_blocks)
        
        # Count for each uncovered t-subset how many blocks can cover it
        cover_counts = np.zeros(len(uncovered_idxs), dtype=np.int32)
        for t_idx in uncovered_idxs:
            for b_idx in t_sub_to_k_idxs[t_idx]:
                if b_idx not in essential_set:
                    cover_counts[t_idx] += 1
        
        # Lower bound: at least max(1, ceil(sum(1/cover_count)))
        # But this is weak. Let's use a better bound.
        
        # Better: The minimum number of blocks needed is at least the maximum over all 
        # t-subsets of (1 / fraction_of_blocks_that_cover_it)
        # Actually, a standard lower bound for set cover is:
        # LB = max over all subsets S of (|S| / max_{b in B} |b ∩ S|)
        # But that's expensive.
        
        # Simpler: LB = sum over t-subsets of 1 / (number of blocks covering it)
        # This is the LP dual bound (weak duality)
        
        valid_counts = cover_counts[cover_counts > 0]
        if len(valid_counts) == 0:
            return 0
            
        # LP dual bound: sum of 1/cover_count for each element
        lb = sum(1.0 / c for c in valid_counts)
        
        # Also: at least the number of uncovered t-subsets divided by the max coverage per block
        max_coverage = 0
        for b_idx in range(n_k):
            if b_idx in essential_set:
                continue
            cov = (block_masks[b_idx] & uncovered_mask).bit_count()
            if cov > max_coverage:
                max_coverage = cov
        
        if max_coverage > 0:
            lb2 = len(uncovered_idxs) / max_coverage
            lb = max(lb, lb2)
            
        return int(np.ceil(lb))

    def branch_and_bound(essential_blocks, uncovered_mask, current_count, best_count, time_limit=10.0):
        """
        Branch and bound to find if we can cover all uncovered t-subsets with fewer than best_count blocks.
        Returns the minimum number of blocks found, or best_count if no improvement.
        """
        if uncovered_mask == 0:
            return current_count
        
        if time.time() > time_limit:
            return best_count
            
        # Lower bound via LP relaxation
        lb = lp_relaxation_lower_bound(essential_blocks, uncovered_mask)
        if current_count + lb >= best_count:
            return best_count
            
        # Find the most constrained uncovered t-subset (fewest covering blocks)
        uncovered_idxs = []
        temp = uncovered_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            uncovered_idxs.append(idx)
            temp ^= lsb
        
        # For each uncovered t-subset, count available blocks
        essential_set = set(essential_blocks)
        best_t_idx = -1
        best_num_candidates = float('inf')
        for t_idx in uncovered_idxs:
            num_cands = 0
            for b_idx in t_sub_to_k_idxs[t_idx]:
                if b_idx not in essential_set:
                    num_cands += 1
            if num_cands < best_num_candidates:
                best_num_candidates = num_cands
                best_t_idx = t_idx
                if num_cands <= 1:
                    break
        
        if best_t_idx == -1 or best_num_candidates == 0:
            return best_count
            
        # Branch on blocks that cover best_t_idx
        candidates = []
        for b_idx in t_sub_to_k_idxs[best_t_idx]:
            if b_idx in essential_set:
                continue
            candidates.append(b_idx)
        
        # Sort by gain (number of new t-subsets covered)
        candidates_with_gain = []
        for b_idx in candidates:
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            candidates_with_gain.append((gain, b_idx))
        
        candidates_with_gain.sort(reverse=True)
        
        # Limit branching to top candidates for efficiency
        max_branch = min(10, len(candidates_with_gain))
        
        result = best_count
        for gain, b_idx in candidates_with_gain[:max_branch]:
            new_essential = essential_blocks + [b_idx]
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            new_count = current_count + 1
            
            if new_count >= result:
                break
                
            # Recurse
            res = branch_and_bound(new_essential, new_uncovered, new_count, result, time_limit)
            if res < result:
                result = res
                if result == current_count + 1:
                    break  # Can't do better than +1
            
            if time.time() > time_limit:
                break
                
        return result

    def local_search_1_for_1(blocks_idx):
        """
        Try to replace blocks with better ones.
        For each block B, find its essential 5-subsets (covered only by B).
        Find a block B' that covers all essential subsets.
        If B' covers more total subsets than B, or if swapping allows further reduction, swap.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Rebuild cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            # Identify essential subsets for each block
            essential_masks = [0] * len(blocks_idx)
            for i, b_idx in enumerate(blocks_idx):
                mask = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        mask |= (1 << t_idx)
                essential_masks[i] = mask
                
            # Try to swap
            for i, b_idx in enumerate(blocks_idx):
                ess_mask = essential_masks[i]
                if ess_mask == 0:
                    continue
                
                # Find candidates that cover ess_mask
                first_ess_t_idx = (ess_mask & -ess_mask).bit_length() - 1
                candidates = t_sub_to_k_idxs[first_ess_t_idx]
                
                best_cand = -1
                best_score = -1
                
                current_score = block_masks[b_idx].bit_count()
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in blocks_idx:
                        continue
                    
                    if (block_masks[cand_idx] & ess_mask) != ess_mask:
                        continue
                    
                    cand_score = block_masks[cand_idx].bit_count()
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_cand = cand_idx
                
                if best_cand != -1:
                    blocks_idx[i] = best_cand
                    improved = True
                    break
            
        return blocks_idx

    def local_search_remove(blocks_idx):
        """
        Try to remove blocks by checking if the remaining blocks still cover everything.
        Uses a more aggressive approach: try removing multiple blocks at once.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        improved = True
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try removing each block
            for i in range(len(blocks_idx)):
                if time.time() >= deadline - 2.0:
                    break
                    
                # Temporarily remove block i
                test_blocks = blocks_idx[:i] + blocks_idx[i+1:]
                
                # Check if test_blocks covers everything
                # Build cover count
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in test_blocks:
                    cover_count[k_sub_t_idxs[b_idx]] += 1
                
                if np.all(cover_count >= 1):
                    blocks_idx = test_blocks
                    improved = True
                    break
        
        return blocks_idx

    def local_search_rebuild(blocks_idx):
        """
        Take the most essential blocks as a core, and re-greedy the rest.
        """
        if not blocks_idx:
            return blocks_idx
            
        blocks_idx = list(blocks_idx)
        
        # Calculate cover counts to find essential blocks
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1
            
        # Score each block by how many essential subsets it has
        block_scores = []
        for b_idx in blocks_idx:
            score = 0
            for t_idx in k_sub_t_idxs[b_idx]:
                if cover_count[t_idx] == 1:
                    score += 1
            block_scores.append((score, b_idx))
            
        # Sort by score descending
        block_scores.sort(reverse=True)
        
        # Keep the top 50% as core
        core_size = max(1, len(blocks_idx) // 2)
        core_blocks = [b for _, b in block_scores[:core_size]]
        core_set = set(core_blocks)
        
        # Determine what is covered by core
        covered_by_core = 0
        for b_idx in core_blocks:
            covered_by_core |= block_masks[b_idx]
            
        # What is still uncovered?
        all_t_mask = (1 << n_t) - 1
        to_cover = all_t_mask & ~covered_by_core
        
        if to_cover == 0:
            return core_blocks
            
        new_blocks = list(core_blocks)
        used = set(core_blocks)
        current_uncovered = to_cover
        
        while current_uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            s_idx = -1
            while s_idx == -1:
                s_idx = random.randint(0, n_t - 1)
                if not ((current_uncovered >> s_idx) & 1):
                    s_idx = -1
                    continue
                break
            
            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in used:
                    continue
                gain = (block_masks[b_idx] & current_uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                return blocks_idx
                
            new_blocks.append(best_block_idx)
            used.add(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            
        return new_blocks

    def branch_and_bound_polish(blocks_idx, time_budget=10.0):
        """
        Use branch and bound to try to reduce the block count.
        Start with the current solution and try to find a smaller one.
        """
        if not blocks_idx:
            return blocks_idx
            
        current_count = len(blocks_idx)
        
        # Try to find a solution with current_count - 1 blocks
        # We'll do this by identifying blocks that might be removable and using B&B
        
        # First, compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        # Find blocks with low essential count
        essential_counts = []
        for b_idx in blocks_idx:
            ess = 0
            for t_idx in k_sub_t_idxs[b_idx]:
                if cover_count[t_idx] == 1:
                    ess += 1
            essential_counts.append((ess, b_idx))
        
        # Sort by essential count ascending (blocks with fewest essentials are most likely removable)
        essential_counts.sort()
        
        # Try removing blocks with 0 essentials first
        removable = [b for ess, b in essential_counts if ess == 0]
        
        if not removable:
            return blocks_idx
        
        # For each removable block, try to see if we can cover its t-subsets with other blocks
        # without adding new blocks
        
        # Use B&B: fix all non-removable blocks, and try to cover the rest with fewer blocks
        fixed_blocks = [b for ess, b in essential_counts if ess > 0]
        fixed_set = set(fixed_blocks)
        
        # What do the fixed blocks cover?
        fixed_cover = 0
        for b_idx in fixed_blocks:
            fixed_cover |= block_masks[b_idx]
        
        # What needs to be covered?
        all_t_mask = (1 << n_t) - 1
        to_cover = all_t_mask & ~fixed_cover
        
        if to_cover == 0:
            # Fixed blocks cover everything, so we can remove all removable blocks
            return fixed_blocks
        
        # Now we need to cover to_cover with blocks from removable (and possibly others)
        # But we want to use fewer than current_count - len(fixed_blocks) blocks
        # i.e., fewer than len(removable) blocks
        
        target = len(removable) - 1  # We want at most this many additional blocks
        
        # Use B&B to find if we can cover to_cover with <= target blocks
        # Start with empty essential set
        best_found = current_count  # Current solution size
        
        # Time limit for this B&B call
        bb_time_limit = min(time_budget, deadline - time.time() - 2.0)
        if bb_time_limit <= 0:
            return blocks_idx
            
        # B&B: try to cover to_cover with at most target blocks
        # We'll use a recursive search
        def bb_search(essential, uncovered, depth, max_depth):
            nonlocal best_found
            
            if uncovered == 0:
                if depth < best_found - len(fixed_blocks):
                    best_found = len(fixed_blocks) + depth
                return True
                
            if depth >= max_depth:
                return False
                
            if time.time() > start_time + (deadline - 2.0):
                return False
                
            # Lower bound
            lb = lp_relaxation_lower_bound(essential, uncovered)
            if depth + lb > max_depth:
                return False
                
            # Find most constrained uncovered t-subset
            uncovered_idxs = []
            temp = uncovered
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                uncovered_idxs.append(idx)
                temp ^= lsb
            
            if not uncovered_idxs:
                return False
                
            essential_set = set(essential)
            best_t_idx = -1
            best_num_cands = float('inf')
            for t_idx in uncovered_idxs:
                num_cands = 0
                for b_idx in t_sub_to_k_idxs[t_idx]:
                    if b_idx not in essential_set and b_idx not in fixed_set:
                        num_cands += 1
                if num_cands < best_num_cands:
                    best_num_cands = num_cands
                    best_t_idx = t_idx
                    if num_cands <= 1:
                        break
            
            if best_t_idx == -1 or best_num_cands == 0:
                return False
                
            candidates = []
            for b_idx in t_sub_to_k_idxs[best_t_idx]:
                if b_idx in essential_set or b_idx in fixed_set:
                    continue
                candidates.append(b_idx)
            
            candidates_with_gain = []
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                candidates_with_gain.append((gain, b_idx))
            
            candidates_with_gain.sort(reverse=True)
            
            max_branch = min(8, len(candidates_with_gain))
            
            for gain, b_idx in candidates_with_gain[:max_branch]:
                new_essential = essential + [b_idx]
                new_uncovered = uncovered & ~block_masks[b_idx]
                
                if bb_search(new_essential, new_uncovered, depth + 1, max_depth):
                    return True
            
            return False
        
        # Try to find a solution with fewer blocks
        success = bb_search([], to_cover, 0, target)
        
        if success and best_found < current_count:
            # We found a better solution, but we need to reconstruct it
            # For now, just return the fixed blocks + greedy fill
            # This is a simplification - in practice we'd track the solution
            pass
        
        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0, randomness=0.0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Randomized Greedy Restarts
    candidates_for_ls = []
    
    for seed in range(1, 100):
        if time.time() >= deadline - 30.0:
            break
        
        rand_val = random.random()
        blocks_idx = greedy_cover(seed, randomness=rand_val)
        blocks_idx = reduce_blocks(blocks_idx)
        
        if blocks_idx:
            candidates_for_ls.append(blocks_idx)
            if len(blocks_idx) < best_len:
                best_blocks = blocks_idx
                best_len = len(blocks_idx)

    # 3. Local Search on Top Candidates
    candidates_for_ls.sort(key=len)
    
    processed = set()
    for blocks_idx in candidates_for_ls[:10]:
        if time.time() >= deadline - 10.0:
            break
        
        h = hash(tuple(sorted(blocks_idx)))
        if h in processed:
            continue
        processed.add(h)
        
        # Apply 1-for-1 local search
        ls_blocks = local_search_1_for_1(blocks_idx)
        ls_blocks = reduce_blocks(ls_blocks)
        
        if ls_blocks and len(ls_blocks) < best_len:
            best_blocks = ls_blocks
            best_len = len(ls_blocks)
            
        # Apply removal local search
        rem_blocks = local_search_remove(blocks_idx)
        rem_blocks = reduce_blocks(rem_blocks)
        
        if rem_blocks and len(rem_blocks) < best_len:
            best_blocks = rem_blocks
            best_len = len(rem_blocks)
            
        # Apply Rebuild Local Search
        rb_blocks = local_search_rebuild(blocks_idx)
        rb_blocks = reduce_blocks(rb_blocks)
        rb_blocks = local_search_1_for_1(rb_blocks)
        rb_blocks = reduce_blocks(rb_blocks)
        
        if rb_blocks and len(rb_blocks) < best_len:
            best_blocks = rb_blocks
            best_len = len(rb_blocks)

    # 4. Final Polish on Best
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search_1_for_1(best_blocks)
        best_blocks = reduce_blocks(best_blocks)
        best_blocks = local_search_remove(best_blocks)
        best_blocks = reduce_blocks(best_blocks)
        
        # Try one more rebuild on the very best
        if time.time() < deadline - 2.0:
            rb_best = local_search_rebuild(best_blocks)
            rb_best = reduce_blocks(rb_best)
            rb_best = local_search_1_for_1(rb_best)
            rb_best = reduce_blocks(rb_best)
            if rb_best and len(rb_best) < len(best_blocks):
                best_blocks = rb_best

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
