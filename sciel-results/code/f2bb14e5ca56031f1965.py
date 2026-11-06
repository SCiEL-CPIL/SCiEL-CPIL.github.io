    def _verify_booleans_numpy(network):
        # ... existing code ...
        
    def construct():
        global _START_TIME
        _START_TIME = time.time()
        
        best = _odd_even_transposition(N)
        if not _verify_booleans_numpy(best):
            return best
            
        current = list(best)
        
        # Helper to check time
        def time_up():
            return (_now() - _START_TIME) >= _DEADLINE

        # Phase 1: Aggressive Greedy Removal
        # We already have this logic, let's integrate it into a loop that allows kicks.
        
        no_improvement_count = 0
        max_no_improvement = 100 # If we try 100 random moves without improvement, maybe restart or kick harder?
        
        while not time_up():
            improved = False
            
            # 1. Try Greedy Removal
            for idx in range(len(current)):
                if time_up():
                    return best
                    
                candidate = current[:idx] + current[idx+1:]
                if _verify_booleans_numpy(candidate):
                    current = candidate
                    if len(current) < len(best):
                        best = list(current)
                        no_improvement_count = 0
                    improved = True
                    break # Restart removal loop
                    
            if improved:
                continue
                
            # 2. If stuck, try random moves to escape local optimum
            # Move 1: Swap two comparators
            # We pick two random indices.
            # If they are independent, swap is always valid.
            # If dependent, we must verify.
            
            if len(current) < 2:
                break
                
            idx1, idx2 = random.sample(range(len(current)), 2)
            
            # Check if independent
            c1 = current[idx1]
            c2 = current[idx2]
            is_independent = (c1[0] not in c2 and c1[1] not in c2)
            
            if is_independent:
                # Swap is safe
                current[idx1], current[idx2] = current[idx2], current[idx1]
                # No need to verify, but we should check if this new arrangement allows removals
                # We loop back to try removals
            else:
                # Dependent swap, need to verify
                candidate = list(current)
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                if _verify_booleans_numpy(candidate):
                    current = candidate
                    # If this swap leads to a removable comparator, great.
                    # We don't update best here unless length decreases (it doesn't in swap)
                    # But it might enable future removals.
                else:
                    # Revert
                    pass
            
            # Move 2: Move a comparator to a different position
            # Pick random idx, random new_idx
            # This is more expensive. Let's stick to swaps for now as they are cheap.
            
            # To prevent infinite loop of useless swaps, we can track state?
            # Or just rely on the time limit.
            
            # Optimization: If we do a swap, we should immediately try to remove again.
            # The loop structure above does this (it loops back to top).
            
            # However, if we keep swapping and never removing, we waste time.
            # We should prioritize removals.
            
            # Let's add a counter for swaps without removal.
            # If we do too many swaps without removal, we might want to try a "kick" (random valid network mutation?)
            # Or just accept that we are stuck.
            
            # Actually, the greedy removal loop iterates through ALL indices.
            # So it's not just trying one random removal. It's trying all.
            # So if we are stuck, it means NO single removal works.
            # So we MUST change the structure (swap/move) to enable a removal.
            
            # So the logic is:
            # 1. Try all removals.
            # 2. If none work, perform a random valid mutation (Swap independent or Verified Swap Dependent).
            # 3. Go back to 1.
            
            # This seems sound.
            
            # One risk: Swapping independent comparators doesn't change the network function.
            # So if removal was impossible before, it's still impossible after swapping independent comparators?
            # Wait. The set of comparators is the same. The order is different.
            # Does the order of independent comparators matter for redundancy?
            # Yes.
            # Example:
            # (0,1) and (2,3). Independent.
            # If we have (0,1), (2,3), (1,2).
            # (0,1) and (2,3) are independent.
            # If we swap them: (2,3), (0,1), (1,2).
            # The function is identical.
            # So if (1,2) was redundant in the first case, it is redundant in the second.
            # If it was NOT redundant in the first, it is NOT redundant in the second.
            # Therefore, swapping independent comparators DOES NOT help with removal!
            
            # CRITICAL INSIGHT: Swapping independent comparators is an isomorphism of the network.
            # It does not change the sorting function.
            # Therefore, if a comparator is not removable in one ordering of independent comparators, it is not removable in any other.
            # So we must swap DEPENDENT comparators to change the network function.
            
            # So the move must be:
            # 1. Pick two dependent comparators.
            # 2. Swap them.
            # 3. Verify.
            # 4. If valid, accept.
            # 5. Try removals.
            
            # This is much more constrained.
            # How many dependent pairs are there?
            # In a dense network, most pairs are dependent.
            
            # Let's implement this.
            
            # We need to ensure we don't just flip-flop.
            # We can keep a history of last K swaps?
            # Or just random.
            
            # Let's refine the loop:
            # while not time_up():
            #    # 1. Greedy Removal
            #    removed = False
            #    for idx in range(len(current)):
            #        ...
            #        if valid:
            #            current = candidate
            #            best = min(best, current)
            #            removed = True
            #            break
            #            
            #    if removed:
            #        continue
            #        
            #    # 2. Random Dependent Swap
            #    if len(current) < 2: break
            #    
            #    # Pick random pair
            #    while True:
            #        i, j = random.sample(range(len(current)), 2)
            #        if current[i] and current[j] are dependent:
            #            break
            #            
            #    candidate = list(current)
            #    candidate[i], candidate[j] = candidate[j], candidate[i]
            #    
            #    if _verify_booleans_numpy(candidate):
            #        current = candidate
            #        # Do not update best (same size)
            #        # But this new state might allow removal.
            #    else:
            #        # Try another swap? Or give up?
            #        # If we fail, we just loop again and try a different swap.
            #        pass

    This looks like a solid plan.
    
    One optimization: The greedy removal loop `for idx in range(len(current))` is O(N^2) in the number of comparators if we verify each time.
    Verification is O(C * 2^N).
    If C=100, 2^N=131k.
    100 * 131k * 17 ops?
    It's fast enough.
    
    However, we can speed up the "Dependent Swap" selection.
    Instead of picking random indices and checking dependency, we can precompute dependencies?
    Or just pick random indices. With 17 wires, probability of independence is low if the network is dense.
    Probability of two random comparators sharing a wire:
    Each comparator uses 2 wires.
    Total wires 17.
    P(shared) = 1 - P(disjoint).
    P(disjoint) = (15 * 14) / (17 * 16) approx?
    If wire 1 is in c1, c2 must not have wire 1 or wire 2 of c1.
    15 wires left for c2's first wire? No, c2 is fixed.
    Let c1 = (a,b).
    c2 = (c,d).
    Disjoint if {c,d} ∩ {a,b} = empty.
    Total pairs (c,d) with c<d from 17 wires: 17*16/2 = 136.
    Pairs intersecting (a,b):
    Total - Disjoint.
    Disjoint pairs chosen from remaining 15 wires: 15*14/2 = 105.
    So P(disjoint) = 105/136 ≈ 0.77.
    So most random pairs are independent!
    We will waste time picking independent pairs and rejecting them.
    
    Better: Pick a random comparator `c1`.
    Pick a random wire `w` in `c1`.
    Find all other comparators that use `w`.
    Pick one randomly.
    This guarantees dependency.
    
    Implementation:
    1.  Pick random index `i`.
    2.  Get `c1 = current[i]`.
    3.  Get wires `w1, w2`.
    4.  Create list of indices `j` such that `j != i` and (`w1` in `current[j]` or `w2` in `current[j]`).
    5.  If list empty, try again (or break).
    6.  Pick random `j` from list.
    7.  Swap `current[i]` and `current[j]`.
    8.  Verify.
    
    This is efficient.

    Let's code this.

    Also, I should add a "kick" if we are stuck for a long time?
    Maybe try to insert a comparator?
    If we are stuck at size K, and no removal works, and no swap works...
    Actually, if no swap works, it means the neighborhood is a local minimum.
    
    I will stick to the Swap strategy.

    One more thing: The seed is 136.
    If we find 80, score is 136/80 = 1.7.
    Inspiration 1 got 1.54 (approx 88 comps).
    We can likely do better.

    I will replace the `construct` function.

    I also need to make sure `_verify_booleans_numpy` is robust.
    The current one creates `wires` array of size (131072, 17).
    This is ~2.2 MB.
    It applies comparators.
    This is fine.

    I will make small edits.

    1. Replace `construct` body.
    2. No other changes needed.

    Wait, `random` module is imported.
    `time` is imported.
    `numpy` is imported.

    The edit:

