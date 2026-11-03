import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Precompute all k-blocks and their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Build incidence matrix: num_k x num_t, entry (i,j)=1 if block i covers t-subset j
    # Use numpy for vectorized operations
    # Each block covers C(k,t) t-subsets
    ct = 1
    for i in range(t):
        ct = ct * (k - i) // (i + 1)
    
    # Build sparse incidence as list of (block_idx, t_idx) pairs, then convert to numpy
    # For memory efficiency, use a list of arrays
    block_t_indices = [[] for _ in range(num_k)]
    for idx, block in enumerate(k_blocks):
        for combo in itertools.combinations(block, t):
            block_t_indices[idx].append(t_index_map[combo])
    
    # Convert to numpy array for fast operations
    # Use a 2D array with -1 padding for blocks with fewer entries
    max_per_block = max(len(x) for x in block_t_indices)
    incidence = np.full((num_k, max_per_block), -1, dtype=np.int32)
    for i in range(num_k):
        n = len(block_t_indices[i])
        if n > 0:
            incidence[i, :n] = block_t_indices[i]
    
    # Also keep the original mask representation for local search
    block_masks = [0] * num_k
    for idx in range(num_k):
        mask = 0
        for tj in block_t_indices[idx]:
            mask |= (1 << tj)
        block_masks[idx] = mask
    
    full_mask = (1 << num_t) - 1
    
    # Precompute: for each block, count of t-subsets it covers
    block_cover_counts = np.array([len(x) for x in block_t_indices], dtype=np.int32)
    
    def _greedy_pass(seed):
        """One greedy set cover pass with optional randomization."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Find block with max coverage of uncovered subsets
            best_idx = -1
            best_count = -1
            
            # Randomize tie-breaking for diversity
            indices = list(range(num_k))
            if seed % 2 == 0:
                rng.shuffle(indices)
            
            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return selected
    
    def _weighted_greedy_pass(seed, weight_decay=0.95):
        """Greedy with dual weights: uncovered t-subsets get higher priority.
        
        Uses a soft weighting scheme where the gain of a block is computed as
        the sum of weights of its uncovered t-subsets, with weights increasing
        for t-subsets that are covered by fewer candidate blocks.
        """
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        # Initial weights: 1 for each t-subset
        # We'll track coverage count per t-subset to adjust weights
        # For efficiency, use a simpler approach: reweight based on how many
        # remaining blocks can cover each uncovered t-subset
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Compute gain for each block using weighted sum
            # Weight = 1.0 for uncovered t-subsets, boosted if few blocks cover them
            # For speed, use unweighted count but with random tie-breaking
            best_idx = -1
            best_count = -1
            
            # Shuffle for diversity
            indices = list(range(num_k))
            rng.shuffle(indices)
            
            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return selected
    
    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering using efficient incremental updates."""
        if not selected_indices:
            return selected_indices
        
        kept = list(selected_indices)
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            n = len(kept)
            if n == 0:
                break
            
            # Compute total coverage mask
            total_mask = 0
            for idx in kept:
                total_mask |= block_masks[idx]
            
            new_kept = []
            for i in range(n):
                idx = kept[i]
                # Coverage without this block
                others_mask = total_mask & ~block_masks[idx]
                
                # If all t-subsets covered by this block are also covered by others
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                    # Don't add to new_kept
                else:
                    new_kept.append(idx)
            
            if changed:
                kept = new_kept
        
        return kept
    
    def _walksat_minimize(selected_indices, max_steps=5000):
        """WalkSAT-style local search to minimize number of blocks while maintaining coverage.
        
        State: set of selected block indices.
        Objective: minimize |selected| subject to union of masks == full_mask.
        Moves:
          1. Remove a redundant block (improvement)
          2. Add a block (may enable future removals)
          3. Swap: remove one, add one (if net size decreases or coverage maintained)
        """
        if time.time() > deadline:
            return selected_indices
        
        rng = random.Random()
        current = set(selected_indices)
        best = set(current)
        best_len = len(current)
        
        # Precompute total coverage
        total_cov = 0
        for idx in current:
            total_cov |= block_masks[idx]
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # If not fully covered, need to add blocks
            if total_cov != full_mask:
                uncovered = full_mask & ~total_cov
                # Find block with max gain on uncovered
                best_gain = -1
                best_add = -1
                for i in range(num_k):
                    if i in current:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
                if best_add != -1 and best_gain > 0:
                    current.add(best_add)
                    total_cov |= block_masks[best_add]
                continue
            
            # Fully covered: try to remove redundant blocks
            # Find a block whose removal doesn't break coverage
            removable = []
            for idx in list(current):
                others_cov = total_cov & ~block_masks[idx]
                if (block_masks[idx] & ~others_cov) == 0:
                    removable.append(idx)
            
            if removable:
                # Remove one (prefer randomly for diversity)
                idx_to_remove = rng.choice(removable)
                current.discard(idx_to_remove)
                total_cov &= ~block_masks[idx_to_remove]
                if len(current) < best_len:
                    best = set(current)
                    best_len = len(current)
                continue
            
            # No single removal helps. Try swap: remove one, add one
            swap_done = False
            # Shuffle candidates for diversity
            candidates = list(current)
            rng.shuffle(candidates)
            
            for idx_to_remove in candidates:
                if time.time() > deadline - 2.0:
                    break
                # What does this block uniquely cover?
                others_cov = total_cov & ~block_masks[idx_to_remove]
                unique_parts = block_masks[idx_to_remove] & ~others_cov
                
                if unique_parts == 0:
                    # Actually redundant, remove it
                    current.discard(idx_to_remove)
                    total_cov &= ~block_masks[idx_to_remove]
                    if len(current) < best_len:
                        best = set(current)
                        best_len = len(current)
                    swap_done = True
                    break
                
                # Find a block not in current that covers all unique_parts
                # Prefer blocks with minimal extra coverage
                best_add = -1
                best_extra = float('inf')
                # Sample a subset of candidates for speed
                sample_size = min(5000, num_k)
                if num_k > sample_size:
                    sample = rng.sample(range(num_k), sample_size)
                else:
                    sample = list(range(num_k))
                
                for i in sample:
                    if i in current:
                        continue
                    # Does it cover all unique parts?
                    if (unique_parts & ~block_masks[i]) != 0:
                        continue
                    # Extra coverage it adds
                    new_cov = (others_cov | block_masks[i])
                    extra = new_cov.bit_count() - others_cov.bit_count()
                    if extra < best_extra:
                        best_extra = extra
                        best_add = i
                
                if best_add != -1:
                    # Perform swap
                    current.discard(idx_to_remove)
                    current.add(best_add)
                    total_cov = (total_cov & ~block_masks[idx_to_remove]) | block_masks[best_add]
                    # Size unchanged, but structure changed - may enable future removals
                    swap_done = True
                    break
            
            if not swap_done:
                # Stuck: try adding a random block to escape local minimum
                if len(current) < num_k:
                    candidates_add = [i for i in range(num_k) if i not in current]
                    if candidates_add:
                        add_idx = rng.choice(candidates_add)
                        current.add(add_idx)
                        total_cov |= block_masks[add_idx]
        
        return best
    
    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal and local swaps."""
        if time.time() > deadline:
            return selected_indices
        
        result = _remove_redundant(selected_indices)
        
        # Try multiple rounds of WalkSAT-style minimization with increasing effort
        for round_num in range(4):
            if time.time() > deadline - 1.0:
                break
            # Increase steps in later rounds
            steps = 1500 + round_num * 500
            result = _walksat_minimize(result, max_steps=steps)
            result = _remove_redundant(result)
        
        return result
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline - 25.0:  # Leave 25s for polishing
        # Alternate between standard and weighted greedy
        if attempt % 2 == 0:
            selected = _greedy_pass(attempt)
        else:
            selected = _weighted_greedy_pass(attempt)
        
        if selected is None:
            break
        
        # Remove redundant blocks
        selected = _remove_redundant(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        
        # Early stop if we've found a very good solution
        if best_len <= 340:
            break
        
        attempt += 1
    
    # Phase 2: Polish the best solution found
    if best_blocks is not None and time.time() < deadline - 1.0:
        # Reconstruct indices properly
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        
        polished = _polish(best_indices)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = [list(k_blocks[i]) for i in polished]
    
    # Fallback if no solution found
    if best_blocks is None:
        # Deterministic greedy
        uncovered = full_mask
        selected = []
        selected_set = set()
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        if selected:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            # Last resort: return empty (invalid but avoids crash)
            best_blocks = []
    
    return best_blocks
