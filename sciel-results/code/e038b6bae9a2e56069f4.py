"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    # Deadline guard: finish well before the 120s wall-clock limit.
    deadline = time.time() + 115.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct product with a single symbol
        # This is not optimal for general cases but keeps the interface valid.
        # For the specific task (q=3, n=6, r=1) we use the specialized logic below.
        # A simple greedy construction for general case:
        codewords = []
        covered = [False] * (q ** n)
        # Map tuple to index
        def tup_to_idx(t):
            idx = 0
            for i in range(n):
                idx = idx * q + t[i]
            return idx
        # Precompute neighbors for speed if possible, but for general q,n it's heavy.
        # We'll just do a basic greedy here for non-target cases.
        for word in itertools.product(range(q), repeat=n):
            w_idx = tup_to_idx(word)
            if not covered[w_idx]:
                codewords.append(list(word))
                # Mark this and all neighbors as covered
                covered[w_idx] = True
                for pos in range(n):
                    for val in range(q):
                        if val == word[pos]:
                            continue
                        new_word = list(word)
                        new_word[pos] = val
                        new_idx = tup_to_idx(tuple(new_word))
                        covered[new_idx] = True
                if time.time() > deadline - 5:
                    break
        return codewords

    # Specialized for q=3, n=6, radius=1
    # Total words: 729
    # Start with the known optimal structure based on tetracode or a strong seed.
    
    # Generate all 729 words as tuples
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1) for each word as bitsets
    # 729 bits fits easily in a Python int.
    ALL_BITS = (1 << 729) - 1
    
    coverage = [0] * 729
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    j = word_to_idx[tuple(new_w)]
                    mask |= (1 << j)
        coverage[i] = mask

    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(tuple(h + list(tail)))
    
    initial_indices = [word_to_idx[w] for w in initial_code]
    
    # Helper to check if a set of indices covers everything
    def is_covered(indices):
        cover = 0
        for idx in indices:
            cover |= coverage[idx]
        return cover == ALL_BITS

    # Helper to get current coverage mask
    def get_cover_mask(indices):
        cover = 0
        for idx in indices:
            cover |= coverage[idx]
        return cover

    # Greedy Reduction using bitsets
    def try_remove_redundant_fast(indices):
        code_set = set(indices)
        current_cover = get_cover_mask(code_set)
        
        changed = True
        while changed:
            changed = False
            # Iterate over a list to allow modification
            for idx in list(code_set):
                # If we remove idx, new cover is current_cover & ~coverage[idx]
                # We need new_cover == ALL_BITS
                if (current_cover & ~coverage[idx]) == ALL_BITS:
                    code_set.remove(idx)
                    current_cover &= ~coverage[idx]
                    changed = True
        
        return list(code_set)

    # Local Search with Swaps
    # Goal: Find a code of size 73 (or smaller).
    # We start with the reduced initial code.
    best_code = try_remove_redundant_fast(initial_indices)
    best_size = len(best_code)
    
    # If we are already at 73, we might try to find 72 (unlikely but let's search)
    # Or just ensure we don't get stuck at 74+.
    
    # Perform a local search:
    # 1. Try removing any word (already done).
    # 2. Try swapping a word: remove u, add v.
    #    Condition: (current_cover & ~coverage[u] | coverage[v]) == ALL_BITS
    #    This keeps size same. Why do this? To change the structure to allow further removals?
    #    Actually, if we are at a local minimum for removal, swapping doesn't reduce size directly.
    #    However, it might move us to a different local minimum from which a removal is possible.
    #    Or, we can try "2-for-1" swaps: remove u, add v1, v2? No, that increases size.
    #    The most effective move for reduction is often: Remove u, Remove w, Add v?
    #    Let's stick to 1-for-1 swaps to explore the space of K-word codes, and then check for removals.
    
    def local_search_swap_remove(indices, max_iters=1000):
        code_set = set(indices)
        current_cover = get_cover_mask(code_set)
        current_size = len(code_set)
        
        # Precompute list of all words not in code_set for random selection?
        # Or just pick random word from 0..728.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try all possible swaps? 73 * 729 is ~50k checks. Fast with bitsets.
            # But we only need to find ONE improvement (a removal enabled by a swap).
            # So: For each u in code_set, for each v not in code_set:
            #   Check if swap is valid.
            #   If valid, perform swap, then try to remove any word.
            #   If a removal happens, accept the new state (size - 1).
            #   If no removal, revert swap.
            
            # Optimization: Only try swaps that might help?
            # Let's just iterate.
            
            u_list = list(code_set)
            random.shuffle(u_list)
            
            for u in u_list:
                if time.time() > deadline - 5:
                    return list(code_set), current_size
                # Try to find a v that allows a removal after swap
                # New cover after swap u->v: cover_swap = (current_cover & ~coverage[u]) | coverage[v]
                # We need cover_swap to be a superset of ALL_BITS (it always is if swap is valid)
                # Valid swap: (current_cover & ~coverage[u]) | coverage[v] == ALL_BITS
                # After swap, we can remove w if (cover_swap & ~coverage[w]) == ALL_BITS
                
                # To speed up, we can precompute the "deficit" or just iterate v.
                # There are 729 - 73 = 656 candidates for v.
                # 73 * 656 = 47k checks.
                # Each check involves a few bit ops. Very fast.
                
                for v in range(729):
                    if v in code_set:
                        continue
                    
                    # Check if swap u->v is valid
                    # Note: current_cover includes coverage[u]
                    # We remove u, add v.
                    # temp_cover = (current_cover & ~coverage[u]) | coverage[v]
                    # If temp_cover != ALL_BITS, skip.
                    # Note: If u and v cover disjoint unique parts, this might fail.
                    
                    temp_cover = (current_cover & ~coverage[u]) | coverage[v]
                    if temp_cover != ALL_BITS:
                        continue
                    
                    # Swap is valid. Now check if we can remove any w from (code_set - {u} + {v})
                    # New set: code_set \ {u} U {v}
                    # We can remove w if (temp_cover & ~coverage[w]) == ALL_BITS
                    # Note: w can be v? No, we just added v.
                    # w can be any other word in the new set.
                    
                    # To be efficient, we can check if there is ANY w in the new set that is redundant.
                    # Redundant w: temp_cover & ~coverage[w] == ALL_BITS
                    
                    # We can iterate w in code_set (excluding u, including v? No, v was just added, 
                    # but if v is redundant immediately, we could have just not added it? 
                    # But v might be needed to cover something that u covered, allowing removal of another w).
                    
                    # Let's just check all w in the new set.
                    # New set size is same.
                    # We can approximate by checking if temp_cover has "slack".
                    # But exact check:
                    
                    # Optimization: If we find a valid swap that allows a removal, we accept it.
                    # We don't need to check all w for all v. We can pick a random v first?
                    # No, deterministic is safer for reproducibility, but random is faster to find *a* move.
                    # Let's shuffle v candidates?
                    # For now, let's just check a few random v's per u to keep it fast?
                    # Or just all. 47k ops is nothing.
                    
                    # Let's try to find a w to remove.
                    # We can iterate w in code_set.
                    # If w == u, we are effectively just adding v (size +1) then removing u (size same)? No.
                    # The move is: Remove u, Add v. Size constant.
                    # Then Remove w. Size -1.
                    # Total: Remove u, Remove w, Add v.
                    
                    # So we are looking for u, w in code_set and v not in code_set such that:
                    # 1. (current_cover & ~coverage[u] & ~coverage[w] | coverage[v]) == ALL_BITS
                    # 2. This implies we can replace {u, w} with {v}.
                    
                    # This is a 2-for-1 move.
                    # This is much more powerful than 1-for-1 swap + remove.
                    # Let's implement 2-for-1 search directly.
                    # For each pair (u, w) in code_set:
                    #   For each v not in code_set:
                    #     Check if (current_cover & ~coverage[u] & ~coverage[w] | coverage[v]) == ALL_BITS
                    #     If yes, replace {u, w} with {v}.
                    
                    # Complexity: C(73, 2) * 656 ~ 2600 * 656 ~ 1.7M checks.
                    # Each check is bit ops. Might be a bit slow in Python if done naively in a tight loop.
                    # 1.7M * ~5 ops = ~10M ops. Should take < 1s.
                    
                    # Let's refactor to 2-for-1 search.
                    pass # We will implement this in the main loop below.
                break # Only try first valid u? No, we want to explore.
            # End u loop
            # If no improvement found in this pass, break?
            # Or continue to try random restarts of the local search?
            # Let's just do one pass of 2-for-1 search.
            
        return list(code_set), current_size

    # Implement 2-for-1 local search
    def local_search_2for1(indices):
        code_set = set(indices)
        current_cover = get_cover_mask(code_set)
        
        # List of indices for iteration
        code_list = list(code_set)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to break symmetry
            random.shuffle(code_list)
            
            # Try all pairs
            for i in range(len(code_list)):
                if time.time() > deadline - 5:
                    break
                u = code_list[i]
                for j in range(i + 1, len(code_list)):
                    w = code_list[j]
                    
                    # Check if we can replace u and w with a single v
                    # Condition: (current_cover & ~coverage[u] & ~coverage[w] | coverage[v]) == ALL_BITS
                    # This is equivalent to:
                    # Let deficit = ALL_BITS & ~(current_cover & ~coverage[u] & ~coverage[w])
                    # We need coverage[v] to cover `deficit`.
                    # i.e., deficit & ~coverage[v] == 0
                    # i.e., (deficit & coverage[v]) == deficit
                    # i.e., deficit is a subset of coverage[v]
                    
                    base_cover = current_cover & ~coverage[u] & ~coverage[w]
                    deficit = ALL_BITS & ~base_cover
                    
                    # If deficit is 0, u and w are both redundant (should have been caught by 1-for-0)
                    # If deficit is not 0, we need a v that covers it.
                    
                    # Find v not in code_set such that coverage[v] covers deficit.
                    # We can iterate v.
                    # Optimization: Precompute for each possible deficit? Too many.
                    # Just iterate v.
                    
                    # To speed up, we can check if any v exists.
                    # We can iterate v in range(729) and check if v not in code_set and (deficit & ~coverage[v]) == 0.
                    
                    # We can also limit v to those that are "good" candidates?
                    # For now, full scan.
                    
                    found_v = -1
                    for v in range(729):
                        if v in code_set:
                            continue
                        if (deficit & ~coverage[v]) == 0:
                            found_v = v
                            break
                    
                    if found_v != -1:
                        # Perform move
                        code_set.remove(u)
                        code_set.remove(w)
                        code_set.add(found_v)
                        current_cover = get_cover_mask(code_set)
                        code_list = list(code_set)
                        improved = True
                        break # Restart loop with new set
                if improved:
                    break
            # End pair loop
        return list(code_set)

    # Run the 2-for-1 local search
    current_code = local_search_2for1(best_code)
    best_size = len(current_code)
    
    # If we are still > 73, try random restarts of the reduction + local search
    # Or just multiple random initial codes?
    # The tetracode seed is strong.
    
    # Let's try a few random perturbations of the best code
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        
        # Perturb: remove one, add one random
        if len(current_code) > 0:
            u = random.choice(current_code)
            v = random.randint(0, 728)
            if v not in current_code:
                perturbed = [x for x in current_code if x != u]
                perturbed.append(v)
                # Check if valid
                if is_covered(perturbed):
                    # Try to reduce
                    reduced = try_remove_redundant_fast(perturbed)
                    # Try 2-for-1
                    reduced = local_search_2for1(reduced)
                    if len(reduced) < len(current_code):
                        current_code = reduced
                        best_size = len(current_code)
    
    # Final check
    if not is_covered(current_code):
        # Fallback
        current_code = initial_indices
        # Try to reduce again just in case
        current_code = try_remove_redundant_fast(current_code)

    # Convert back to lists of integers
    return [list(all_words[idx]) for idx in current_code]

# EVOLVE-BLOCK-END
