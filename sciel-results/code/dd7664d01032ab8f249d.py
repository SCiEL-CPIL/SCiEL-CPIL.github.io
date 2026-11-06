import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # Compute masks: for each block, which t-subsets does it cover?
    # We store these as Python ints for the initial construction, but we will
    # also create a NumPy representation for fast vectorized gain calculation.
    block_data = []
    block_masks_list = []
    for block in all_k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))
        block_masks_list.append(mask)
    
    block_masks = block_masks_list
    full_mask = (1 << n_t) - 1
    
    # Precompute bit counts for all block masks (constant per block)
    block_bit_counts = [m.bit_count() for m in block_masks]
    
    # Prepare NumPy arrays for vectorized gain calculation
    # Split masks into 64-bit chunks
    num_chunks = (n_t + 63) // 64
    block_masks_np = np.zeros((n_k, num_chunks), dtype=np.uint64)
    for i, m in enumerate(block_masks):
        for c in range(num_chunks):
            block_masks_np[i, c] = (m >> (c * 64)) & ((1 << 64) - 1)
    
    def get_gain_vectorized(uncovered_np):
        """Calculate gain for all blocks given an uncovered mask (numpy array of uint64 chunks)."""
        # Perform bitwise AND and sum bits
        # Note: np.uint64 sum might overflow if gain is large, but max gain is C(7,4)=35, so it's safe.
        # Actually, we just need the count of set bits in the intersection.
        # We can use np.bitwise_and then sum.
        # But np.sum on uint64 gives uint64. We need int.
        # Let's use a helper to count bits in uint64 array.
        # Since max gain is small, we can just sum the bits.
        # Wait, np.sum of bits is not direct.
        # We can do: (block_masks_np & uncovered_np[None, :]).sum(axis=1)
        # This sums the values, not the bits.
        # We need to count bits.
        # We can unpack or use a lookup table? No, 64-bit lookup is too big.
        # However, we can use the property that we only care about the *count*.
        # Let's stick to Python loops for gain if vectorization is tricky, OR
        # use a clever trick.
        
        # Actually, for n_k=50k, a Python loop taking ~0.5s is acceptable if it saves 1s in setup or allows more ILS.
        # But let's try to optimize the Python loop first.
        # The current loop iterates n_k times.
        
        # Let's try a different vectorization:
        # We can't easily count bits in a vectorized way in pure NumPy without ufunc.
        # However, we can use `np.unpackbits`? That converts to 0/1 array of size 8*64*num_chunks. Too big.
        
        # Let's optimize the Python loop by using local variables and avoiding function calls.
        return None # Placeholder, we will implement a fast Python loop

    def greedy_construct(use_log=False, rng=None):
        """Greedy set cover with optional log-gain, using a boolean array for remaining."""
        uncovered = full_mask
        selected = []
        remaining = [True] * n_k
        
        while uncovered != 0:
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_score = -1
            # Optimize inner loop
            bm = block_masks
            unc = uncovered
            for i in range(n_k):
                if not remaining[i]:
                    continue
                gain = (bm[i] & unc).bit_count()
                if gain == 0:
                    continue
                if use_log:
                    already_covered = block_bit_counts[i] - gain
                    score = gain / (1 + already_covered * 0.1)
                else:
                    score = gain
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~bm[best_idx]
            remaining[best_idx] = False
        
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks iteratively."""
        if not selected_indices:
            return selected_indices
        
        # Build cover counts
        cover_count = [0] * n_t
        for idx in selected_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
        
        # Verify full coverage
        for i in range(n_t):
            if cover_count[i] == 0:
                return selected_indices
        
        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                # Check if this block is the only cover for any t-subset
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if is_critical:
                    new_current.append(idx)
                else:
                    # Remove this block
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True
            
            current = new_current
        
        return current
    
    def local_search(selected_indices, max_time=30):
        """Optimized local search with faster swap evaluation using precomputed candidate lists."""
        if not selected_indices:
            return selected_indices
        
        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        # Build coverage counts for current solution
        cov = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        # Try deletion pass
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            removed_idx = current[i]
            mask = block_masks[removed_idx]
            
            can_remove = True
            m = mask
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                if cov[k2] == 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                m = mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] -= 1
                    m ^= lsb
                current.pop(i)
                current_set.discard(removed_idx)
                best = current[:]
                best_len = len(best)
                i -= 1
        
        # Try swap: remove one, add one - optimized
        if time.time() < deadline - 20:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                removed_idx = current[i]
                mask = block_masks[removed_idx]
                
                # Temporarily remove
                m = mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] -= 1
                    m ^= lsb
                
                # Find uncovered
                uncovered_mask = 0
                for k2 in range(n_t):
                    if cov[k2] == 0:
                        uncovered_mask |= (1 << k2)
                
                if uncovered_mask == 0:
                    current.pop(i)
                    current_set.discard(removed_idx)
                    best = current[:]
                    best_len = len(best)
                    i -= 1
                else:
                    # Try to find a replacement - only check blocks that cover at least one uncovered subset
                    best_add = -1
                    best_add_gain = -1
                    target_count = uncovered_mask.bit_count()
                    
                    # Check all blocks but skip those in current_set
                    for j in range(n_k):
                        if j in current_set:
                            continue
                        gain = (block_masks[j] & uncovered_mask).bit_count()
                        if gain > best_add_gain:
                            best_add_gain = gain
                            best_add = j
                        if gain == target_count:
                            break
                    
                    if best_add >= 0 and best_add_gain > 0:
                        current[i] = best_add
                        current_set.discard(removed_idx)
                        current_set.add(best_add)
                        m = block_masks[best_add]
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
                    else:
                        m = mask
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
        
        # Try 2-exchange: remove two, add one (net -1)
        if time.time() < deadline - 15 and len(current) >= 2:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                for j in range(i+1, len(current)):
                    if time.time() > deadline - 15:
                        break
                    # Temporarily remove both
                    m1 = block_masks[current[i]]
                    m2 = block_masks[current[j]]
                    for m in (m1, m2):
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            k2 = lsb.bit_length() - 1
                            cov[k2] -= 1
                            mm ^= lsb
                    
                    uncovered_mask = 0
                    for k2 in range(n_t):
                        if cov[k2] == 0:
                            uncovered_mask |= (1 << k2)
                    
                    if uncovered_mask == 0:
                        # Can remove both
                        current.pop(j)
                        current.pop(i)
                        best = current[:]
                        best_len = len(best)
                        break
                    else:
                        # Try to find a replacement that covers all uncovered
                        target_count = uncovered_mask.bit_count()
                        best_add = -1
                        for k in range(n_k):
                            if k in current_set:
                                continue
                            gain = (block_masks[k] & uncovered_mask).bit_count()
                            if gain == target_count:
                                best_add = k
                                break
                        
                        if best_add >= 0:
                            current[i] = best_add
                            current_set.discard(current[j])
                            current_set.add(best_add)
                            current.pop(j)
                            m = block_masks[best_add]
                            while m:
                                lsb = m & -m
                                k2 = lsb.bit_length() - 1
                                cov[k2] += 1
                                m ^= lsb
                            best = current[:]
                            best_len = len(best)
                        else:
                            # Restore
                            for m in (m1, m2):
                                mm = m
                                while mm:
                                    lsb = mm & -mm
                                    k2 = lsb.bit_length() - 1
                                    cov[k2] += 1
                                    mm ^= lsb
        
        best = prune_blocks(best)
        return best
    
    # Phase 1: Multiple greedy constructions with different randomizations
    best_solution = None
    best_len = float('inf')
    
    # First, a deterministic greedy
    sol = greedy_construct(use_log=False)
    sol = prune_blocks(sol)
    sol = local_search(sol)
    best_solution = sol
    best_len = len(sol)
    
    # Try log-gain greedy
    if time.time() < deadline - 40:
        sol = greedy_construct(use_log=True)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Try a third greedy variant: weight by uncovered subset frequency
    if time.time() < deadline - 40:
        # Build frequency of how many blocks cover each t-subset
        sub_freq = [0] * n_t
        for idx in range(n_k):
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                sub_freq[k2] += 1
                m ^= lsb
        
        # Greedy with weighted gain: prefer blocks covering rare t-subsets
        uncovered = full_mask
        selected = []
        remaining = [True] * n_k
        
        while uncovered != 0:
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_score = -1.0
            for i in range(n_k):
                if not remaining[i]:
                    continue
                gain_mask = block_masks[i] & uncovered
                gain = gain_mask.bit_count()
                if gain == 0:
                    continue
                # Weight by inverse frequency of covered subsets
                weight_sum = 0
                m = gain_mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    weight_sum += 1.0 / (1 + sub_freq[k2] * 0.01)
                    m ^= lsb
                score = weight_sum
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            remaining[best_idx] = False
        
        sol = prune_blocks(selected)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Phase 2: ILS - Iterated Local Search
    rng = random.Random(42)
    ils_iter = 0
    while time.time() < deadline - 25 and ils_iter < 120:
        ils_iter += 1
        
        # Perturbation: randomly modify the current solution
        perturbed = best_solution[:]
        perturbed_set = set(perturbed)
        if len(perturbed) > 2:
            # Remove 1-3 random blocks
            n_remove = min(rng.randint(1, 3), len(perturbed) - 1)
            to_remove = rng.sample(range(len(perturbed)), n_remove)
            to_remove_set = set(to_remove)
            for i in sorted(to_remove_set, reverse=True):
                perturbed_set.discard(perturbed[i])
                perturbed.pop(i)
        
        # Re-optimize: add blocks to cover uncovered, then prune, then local search
        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        # Find uncovered
        uncovered = 0
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)
        
        # Greedily add blocks to cover uncovered
        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed_set:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1 or best_gain == 0:
                break
            perturbed.append(best_idx)
            perturbed_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        # Prune and local search
        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed)
        
        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)
    
    # Final validation and return
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb
    
    # If not fully covered (shouldn't happen), add blocks
    sol_set = set(best_solution)
    for k2 in range(n_t):
        if final_cov[k2] == 0:
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in sol_set:
                        best_solution.append(j)
                        sol_set.add(j)
                    break
    
    blocks = [list(block_data[i][0]) for i in best_solution]
    
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    
    return unique_blocks
