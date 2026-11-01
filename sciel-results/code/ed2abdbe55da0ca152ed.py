import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute coverage mask for each k-subset as big-int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask
    
    full_mask = (1 << n_tsubs) - 1
    
    def greedy_cover(seed=None):
        """Greedy set cover with optional random tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        remaining = n_tsubs
        selected = []
        used = set()
        
        while remaining > 0 and time.time() < deadline - 30.0:
            best_j = -1
            best_gain = 0
            best_tiebreak = 0
            
            for j in range(n_ksubs):
                if j in used:
                    continue
                common = uncovered & ksub_masks[j]
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_tiebreak = random.random()
                elif gain == best_gain and gain > 0:
                    if random.random() > best_tiebreak:
                        best_j = j
                        best_tiebreak = random.random()
            
            if best_j == -1 or best_gain == 0:
                break
            
            selected.append(best_j)
            used.add(best_j)
            uncovered &= ~ksub_masks[best_j]
            remaining -= best_gain
        
        return selected, uncovered
    
    def remove_redundant(selected_indices):
        """Remove redundant blocks from a selection."""
        if not selected_indices:
            return selected_indices
        
        active = list(selected_indices)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active):
                other_mask = 0
                for j_idx, bj in enumerate(active):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                if (ksub_masks[bi] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(bi)
            active = new_active
        return active
    
    def local_search(selected_indices, max_iters=None):
        """Local search: try to reduce block count via swaps and removals."""
        if not selected_indices:
            return selected_indices
        
        if max_iters is None:
            max_iters = 200
        
        active = list(selected_indices)
        current_size = len(active)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try removing one block at a time
            for i_idx in range(len(active)):
                if time.time() > deadline - 5.0:
                    break
                bi = active[i_idx]
                other_mask = 0
                for j_idx, bj in enumerate(active):
                    if j_idx != i_idx:
                        other_mask |= ksub_masks[bj]
                if (ksub_masks[bi] & ~other_mask) == 0:
                    active.pop(i_idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try replacing two blocks with one
            for i_idx in range(len(active)):
                if time.time() > deadline - 5.0:
                    break
                for j_idx in range(i_idx + 1, len(active)):
                    if time.time() > deadline - 5.0:
                        break
                    bi = active[i_idx]
                    bj = active[j_idx]
                    # Combined coverage of these two blocks
                    combined = ksub_masks[bi] | ksub_masks[bj]
                    # Other blocks' coverage
                    other_mask = 0
                    for l_idx, bl in enumerate(active):
                        if l_idx != i_idx and l_idx != j_idx:
                            other_mask |= ksub_masks[bl]
                    # What only bi and bj cover (not covered by others)
                    unique_to_pair = combined & ~other_mask
                    # Find a single block that covers all of unique_to_pair
                    best_replace = -1
                    best_gain = 0
                    for j in range(n_ksubs):
                        if j in active:
                            continue
                        common = ksub_masks[j] & unique_to_pair
                        gain = common.bit_count()
                        if gain == unique_to_pair.bit_count():
                            # This block covers everything unique to the pair
                            # Check if it also doesn't break anything
                            # Actually, we need to verify the full coverage
                            new_active = [x for x in active if x not in (bi, bj)]
                            new_active.append(j)
                            # Verify coverage
                            total = 0
                            for idx in new_active:
                                total |= ksub_masks[idx]
                            if total == full_mask:
                                active = new_active
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try single block replacement: replace one block with another
                # that allows removal of another block
                for i_idx in range(len(active)):
                    if time.time() > deadline - 5.0:
                        break
                    bi = active[i_idx]
                    for j in range(n_ksubs):
                        if j in active:
                            continue
                        # Try replacing bi with j
                        new_active = active[:i_idx] + [j] + active[i_idx+1:]
                        # Check if this allows removing another block
                        total_with_j = 0
                        for idx in new_active:
                            total_with_j |= ksub_masks[idx]
                        if total_with_j != full_mask:
                            continue
                        # Try removing each other block
                        for r_idx in range(len(new_active)):
                            if r_idx == i_idx:
                                continue
                            br = new_active[r_idx]
                            other_mask = 0
                            for l_idx, bl in enumerate(new_active):
                                if l_idx != r_idx:
                                    other_mask |= ksub_masks[bl]
                            if (ksub_masks[br] & ~other_mask) == 0:
                                # Remove br
                                active = new_active[:r_idx] + new_active[r_idx+1:]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return active
    
    # Main strategy: multiple greedy restarts with local search
    best_solution = None
    best_size = float('inf')
    
    # First, do a deterministic greedy
    selected, uncovered = greedy_cover(seed=None)
    if uncovered == 0:
        selected = remove_redundant(selected)
        selected = local_search(selected)
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected
    
    # Now do randomized restarts
    attempt = 0
    while time.time() < deadline - 10.0:
        attempt += 1
        selected, uncovered = greedy_cover(seed=attempt)
        if uncovered != 0:
            # Greedy failed to cover all - skip
            continue
        
        selected = remove_redundant(selected)
        selected = local_search(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected
    
    if best_solution is None:
        # Fallback: return empty or try one more time
        selected, uncovered = greedy_cover(seed=42)
        if uncovered == 0:
            selected = remove_redundant(selected)
            best_solution = selected
    
    if best_solution is None:
        return []
    
    # Convert indices to block lists
    blocks = [list(all_ksubs[idx]) for idx in best_solution]
    
    # Final verification
    final_mask = 0
    for idx in best_solution:
        final_mask |= ksub_masks[idx]
    if final_mask != full_mask:
        # Should not happen, but if it does, fall back to greedy-only
        selected, uncovered = greedy_cover(seed=None)
        if uncovered == 0:
            blocks = [list(all_ksubs[idx]) for idx in selected]
        else:
            blocks = []
    
    return blocks
