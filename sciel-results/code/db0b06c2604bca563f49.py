import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill (actually 60s per contract, but be safe)
    
    # Generate all 4-subsets of [16] as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    n = len(all_masks)
    
    # Pre-compute mask to elements for output
    mask_to_elems = {m: tuple(i+1 for i in range(16) if (m >> i) & 1) for m in all_masks}
    
    best_family = []
    best_size = 0
    
    # Helper: Check if a new mask 'new_m' can be added to a family represented by masks list
    # Returns True if valid, False if it creates a sunflower
    def can_add(new_m, family_masks):
        # Group existing family members by their intersection with new_m
        # If two members A, B in family have A & B == (A & new_m) == (B & new_m), 
        # then {A, B, new_m} is a sunflower.
        # Note: A & new_m == B & new_m implies A & B & new_m == A & new_m == B & new_m.
        # We need A & B == A & new_m.
        
        groups = {}
        for A in family_masks:
            inter = A & new_m
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(A)
        
        for inter, members in groups.items():
            k = len(members)
            if k < 2:
                continue
            # Check if any pair in members has intersection == inter
            # Optimization: if k is small, check all pairs
            for i in range(k):
                A = members[i]
                for j in range(i+1, k):
                    B = members[j]
                    if (A & B) == inter:
                        return False
        return True

    # Strategy:
    # 1. Greedy construction with random restarts (fast, gets a baseline)
    # 2. Local search / Hill climbing on the best family found so far to improve it
    #    Moves: 
    #    a) Add a new set if possible
    #    b) Remove one set, then try to add a different one (swap)
    #    c) Remove one set, then try to add two different ones (net gain +1) - might be too slow
    #    Let's stick to (a) and (b) for speed, and maybe try to add multiple at once.

    # Phase 1: Greedy Restart
    while time.time() < deadline - 10.0: # Leave 10s for local search on best
        order = list(range(n))
        random.shuffle(order)
        
        family = []
        fam_set = set()
        
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            S = all_masks[idx]
            
            # Check if S can be added to family
            if can_add(S, family):
                family.append(S)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            best_fam_set = fam_set.copy()
            
        # Early exit if we hit theoretical max or close to it? Max is 250.
        if best_size >= 250:
            break

    # Phase 2: Local Search on the best family found
    # We want to maximize size.
    # Current best: best_family (list of masks)
    
    # To make local search efficient, we need to be able to check validity quickly.
    # The can_add function is O(N^2) in the worst case for a single add.
    # With N~200, N^2 = 40k ops. 
    # If we do 1000 swaps, that's 40M ops. In Python, this might take a few seconds.
    # Let's try to do a simple hill climb:
    # 1. Try to add any unused set.
    # 2. If no add possible, try swaps: remove one, add one.
    #    Actually, "remove one, add one" keeps size same. We want to INCREASE size.
    #    So we need "remove one, add two" (net +1) or just "add one".
    #    If we are stuck in a local maximum where no single add is possible, we might need to remove one to make room for two.
    #    This is a 1-exchange move for size +1.
    
    # Let's implement a loop that tries to improve best_family.
    current_family = best_family[:]
    current_size = best_size
    improved = True
    search_deadline = deadline - 2.0
    
    # To speed up "try to add", we can pre-compute conflicts? No, it's dynamic.
    # Let's just try random adds and swaps.
    
    # Optimization: Keep a list of "blocked" pairs? Too complex.
    # Simple approach:
    # While time remains:
    #   1. Try to add a random unused set. If success, update best.
    #   2. If no add found after some tries, try a 1-for-2 swap.
    #      Pick a random member to remove.
    #      Try to add two random unused sets that are compatible with each other and the rest.
    #      This is expensive.
    
    # Alternative: 
    # The greedy method is quite strong for this problem size.
    # The main issue is getting stuck in local optima.
    # Let's try a "simulated annealing" like approach or just more aggressive random restarts 
    # with a bias towards sets that are "diverse".
    
    # Actually, let's look at Inspiration 3's constraint approach. It was faster?
    # Inspiration 3 scored 0.79, Current 0.77.
    # The current program uses grouping by intersection.
    
    # Let's try to improve the local search by trying to add sets in a specific order.
    # After the initial greedy, we have a family F.
    # We try to add sets from the complement in a random order.
    # If we can't add any, we try to swap.
    
    # Let's implement a simple improvement loop:
    while time.time() < search_deadline:
        # Try to add a set
        added = False
        # Shuffle unused indices
        unused = [i for i in range(n) if i not in best_fam_set]
        random.shuffle(unused)
        
        for idx in unused:
            if time.time() > search_deadline:
                break
            S = all_masks[idx]
            if can_add(S, best_family):
                best_family.append(S)
                best_fam_set.add(idx)
                best_size += 1
                added = True
                if best_size >= 250:
                    break
                # Restart the unused check? Or just continue?
                # Continuing is fine, we just found one.
        
        if best_size >= 250:
            break
            
        if not added:
            # Try a swap: remove one, try to add two.
            # This is complex. Let's try a simpler "kick":
            # Remove a random set, then run a few greedy steps to see if we can get back and more.
            # This is risky but might escape local optima.
            
            # Let's try: Remove 1 random set.
            # Then try to add sets greedily from the unused pool.
            # If we end up with size > current, keep it. Else revert.
            
            if best_size > 0:
                # Pick a random index to remove
                rem_idx = random.randint(0, len(best_family)-1)
                removed_mask = best_family[rem_idx]
                removed_orig_idx = None
                # We need to know which original index this mask corresponds to to remove from set
                # Since masks are unique, we can find it.
                for i in range(n):
                    if all_masks[i] == removed_mask:
                        removed_orig_idx = i
                        break
                
                # Save state
                old_family = best_family[:]
                old_set = best_fam_set.copy()
                old_size = best_size
                
                # Perform removal
                best_family.pop(rem_idx)
                if removed_orig_idx is not None:
                    best_fam_set.remove(removed_orig_idx)
                best_size -= 1
                
                # Try to add sets greedily
                unused = [i for i in range(n) if i not in best_fam_set]
                random.shuffle(unused)
                
                for idx in unused:
                    if time.time() > search_deadline:
                        break
                    S = all_masks[idx]
                    if can_add(S, best_family):
                        best_family.append(S)
                        best_fam_set.add(idx)
                        best_size += 1
                        if best_size > old_size:
                            # We are ahead, keep going? Or break to lock in gain?
                            # Let's keep going to maximize.
                            pass
                        if best_size >= 250:
                            break
                
                # If we didn't improve or got worse, revert?
                # Actually, if we are smaller, we should revert to the best known.
                # But we track "best" globally.
                # Let's just update best if we improved.
                # If we didn't improve, we should probably revert to the previous best state 
                # to avoid degrading the search state for future kicks.
                if best_size <= old_size:
                    best_family = old_family
                    best_fam_set = old_set
                    best_size = old_size
                # If best_size > old_size, we have a new best. The variables are already updated.

    # Final check
    if best_size == 0:
        # Fallback: just return empty or a single set?
        # The problem asks for maximizing, so empty is valid if no set fits?
        # But a single set always fits.
        if n > 0:
            best_family = [all_masks[0]]
            best_size = 1

    return [list(mask_to_elems[m]) for m in best_family]
