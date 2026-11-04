"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_start_time = time.time()
_DEADLINE = 115.0  # Safety margin before 120s kill (evaluator limit)


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _construct_bitmasks(q, n, radius):
    """
    Precompute the coverage bitmask for every word in the space {0..q-1}^n.
    Returns a list of Python integers, where bit i is set if word i is covered.
    """
    space_size = q ** n
    masks = [0] * space_size
    
    # Precompute index for each word to speed up neighbor lookup
    # We can iterate through all words and compute their mask
    for idx in range(space_size):
        # Convert index to word
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        
        mask = 0
        # The word itself
        mask |= (1 << idx)
        
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Calculate new index
                # new_idx = idx + (new_val - orig_val) * q^(n-1-pos)
                # This is faster than converting back and forth
                diff = new_val - orig_val
                power = q ** (n - 1 - pos)
                new_idx = idx + diff * power
                mask |= (1 << new_idx)
        
        masks[idx] = mask
        
        if time.time() > _DEADLINE - 10:
            # If we run out of time precomputing, we can't use this method effectively.
            # But 6561 words * 25 neighbors is ~160k operations, very fast.
            pass
            
    return masks


def _greedy_cover_bitset(q, n, radius, masks, deadline):
    """
    Greedy set cover using bitmasks.
    """
    space_size = q ** n
    all_ones = (1 << space_size) - 1
    covered = 0
    selected_indices = []
    
    # We need to find the word that covers the most uncovered bits.
    # To do this efficiently, we can just iterate through all candidates.
    # 6561 candidates * bit_count operation is fast.
    
    while covered != all_ones:
        best_idx = -1
        best_gain = 0
        
        # Optimization: If we have many candidates, we can sample or use a heap.
        # But for 6561, linear scan is ~6561 * 1us = 6ms per step.
        # 500 steps * 6ms = 3s. This is acceptable.
        
        # Further optimization: Skip indices that are already selected? 
        # No, a selected index might still cover new bits if we didn't pick it first? 
        # No, greedy picks max gain. Once picked, its gain for *future* steps is 0 
        # relative to itself, but it might cover bits that *other* candidates also cover.
        # Actually, if we pick idx, `covered` updates. 
        # We should not pick the same idx twice.
        
        for i in range(space_size):
            if i in selected_indices: # O(N) check, slow. Use a boolean array.
                continue
            
            # Calculate gain: bits in masks[i] that are NOT in covered
            # gain = (masks[i] & ~covered).bit_count()
            # ~covered creates a huge integer. 
            # It's better to do: total_bits_in_mask = masks[i].bit_count()
            # overlap = (masks[i] & covered).bit_count()
            # gain = total_bits_in_mask - overlap
            # But we don't have total_bits precomputed.
            # Let's just use the subtraction method on the bitset.
            
            # Python integers are arbitrary precision, so ~covered works.
            # However, creating ~covered every time is expensive.
            # Let's maintain `uncovered_mask = all_ones ^ covered`.
            
            gain = (masks[i] & (all_ones ^ covered)).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        covered |= masks[best_idx]
        
        if time.time() > deadline - 5:
            break
            
    return selected_indices


def _prune_bitset(q, n, radius, masks, selected_indices, deadline):
    """
    Iteratively remove redundant codewords.
    A codeword i is redundant if masks[i] is a subset of the union of all other selected masks.
    """
    if not selected_indices:
        return selected_indices
        
    selected_set = set(selected_indices)
    
    # Initial pass: compute union of all
    # We will iterate and try to remove each one.
    
    changed = True
    while changed:
        changed = False
        new_selected = []
        
        # To check if i is redundant, we need the union of all j != i in current list.
        # This is O(N^2) if we recompute union every time.
        # N is ~500. N^2 = 250,000. Bit ops are fast.
        
        # Optimization: 
        # 1. Compute total_union of all selected.
        # 2. For each i, check if masks[i] & ~total_union == 0? No, that checks if i covers anything outside total_union.
        #    We want to know if i is covered by others.
        #    i is redundant if masks[i] is subset of (total_union ^ masks[i])? No.
        #    i is redundant if masks[i] & ~(union_of_others) == 0.
        #    union_of_others = total_union & ~masks[i] is NOT correct if masks overlap.
        #    union_of_others is simply the union of all masks except i.
        
        # Let's just compute union_of_others for each i in a loop.
        # Since N is small, this is fine.
        
        for i in selected_indices:
            if time.time() > deadline - 2:
                break
                
            # Compute union of all other masks
            union_others = 0
            for j in selected_indices:
                if i == j:
                    continue
                union_others |= masks[j]
                
            # Check if i is redundant
            if (masks[i] & ~union_others) == 0:
                # Redundant, do not add to new_selected
                changed = True
            else:
                new_selected.append(i)
                
        selected_indices = new_selected
        if time.time() > deadline - 2:
            break
            
    return selected_indices


def _local_search_swap(q, n, radius, masks, selected_indices, deadline):
    """
    Try to reduce size by 1-exchange or 2-exchange? 
    1-exchange: Remove i, add j. Size same. Not useful for minimization unless we can then prune more.
    Actually, we can try to find a set of 2 words that can be replaced by 1 word.
    This is complex. Let's stick to pruning.
    """
    return selected_indices


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    deadline = _DEADLINE
    
    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        # Generic greedy fallback
        return _generic_greedy(q, n, radius, deadline)

    # 1. Precompute masks
    masks = _construct_bitmasks(q, n, radius)
    
    # 2. Greedy Cover
    # We can start with empty set or tetracode.
    # Tetracode is 729 words. Greedy from scratch might be smaller or larger.
    # Usually greedy from scratch is quite good for covering codes.
    
    selected = _greedy_cover_bitset(q, n, radius, masks, deadline)
    
    # 3. Prune
    selected = _prune_bitset(q, n, radius, masks, selected, deadline)
    
    # 4. Verify Coverage
    space_size = q ** n
    all_ones = (1 << space_size) - 1
    covered = 0
    for idx in selected:
        covered |= masks[idx]
        
    if covered != all_ones:
        # If greedy failed to cover (timeout or bug), fallback to tetracode
        tc = _tetracode_solution(q, n, radius)
        if tc:
            # Verify tetracode covers (it should)
            return tc
        return []
    
    # Convert indices to words
    result = []
    for idx in selected:
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result


def _generic_greedy(q, n, radius, deadline):
    """
    Fallback greedy for other parameters.
    """
    space_size = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    def get_neighbors(w):
        nbrs = [w]
        for i in range(n):
            for v in range(q):
                if v == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = v
                nbrs.append(tuple(new_w))
        return nbrs
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        neighbors[w] = get_neighbors(w)
        
    covered = set()
    selected = []
    
    while time.time() < deadline - 5:
        best_word = None
        best_gain = 0
        for w in all_words:
            if w in selected:
                continue
            gain = len(neighbors[w] - covered)
            if gain > best_gain:
                best_gain = gain
                best_word = w
        if best_word is None or best_gain == 0:
            break
        selected.append(best_word)
        covered.update(neighbors[best_word])
        
    # Prune
    changed = True
    while changed:
        changed = False
        new_selected = []
        for w in selected:
            # Check if w is redundant
            # Union of neighbors of others
            union_others = set()
            for other in selected:
                if other == w:
                    continue
                union_others.update(neighbors[other])
            if neighbors[w].issubset(union_others):
                changed = True
            else:
                new_selected.append(w)
        selected = new_selected
        
    return [list(w) for w in selected]

# EVOLVE-BLOCK-END
