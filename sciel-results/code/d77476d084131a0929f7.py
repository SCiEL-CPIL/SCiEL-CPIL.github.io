<<<<<<< SEARCH
def _local_search(result, selected_mask, deadline, max_iters=10000):
    """
    Improve by:
    1. Trying to add a random triple.
    2. If blocked, try to remove one triple that is part of a blocking witness, then add the candidate.
    3. Try "remove 1, add 2" moves to increase size.
    """
    best_result = result[:]
    best_mask = selected_mask
    best_len = len(result)
    
    # Precompute list of indices for random access
    indices = list(range(len(result)))
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
            
        # Strategy 1: Add a random triple
        if random.random() < 0.5:
            cand = random.randrange(N)
            if not (selected_mask >> cand) & 1:
                if _can_add(selected_mask, cand):
                    # Success
                    selected_mask |= (1 << cand)
                    result.append(list(ALL_TRIPLES[cand]))
                    best_result = result[:]
                    best_mask = selected_mask
                    best_len += 1
                else:
                    # Blocked. Find a witness that is fully present.
                    # Pick a random witness that is full, remove a random triple from it.
                    blocked_witnesses = [w for w in WITNESSES[cand] if (selected_mask & w) == w]
                    if blocked_witnesses:
                        w = random.choice(blocked_witnesses)
                        # Pick a random bit set in w
                        # w is a bitmask of 7 triples
                        # We need to find an index j such that (w >> j) & 1 is 1
                        # And j is in our current result (which it must be, since w is subset of selected_mask)
                        # To remove it, we need to find its position in the `result` list.
                        # This is slow if we do it by searching. 
                        # Optimization: Keep a map from triple_idx to position in result? 
                        # Or just remove by value.
                        
                        # Let's find a triple in w to remove.
                        # Iterate through bits of w
                        bits = []
                        tmp_w = w
                        while tmp_w:
                            lsb = tmp_w & -tmp_w
                            bits.append(lsb.bit_length() - 1)
                            tmp_w ^= lsb
                        
                        rem_idx = random.choice(bits)
                        rem_triple = ALL_TRIPLES[rem_idx]
                        
                        # Remove rem_triple from result and selected_mask
                        # Find position
                        pos = -1
                        for i, t in enumerate(result):
                            if t == list(rem_triple):
                                pos = i
                                break
                        if pos != -1:
                            result.pop(pos)
                            selected_mask &= ~(1 << rem_idx)
                            
                            # Now try to add cand
                            if _can_add(selected_mask, cand):
                                selected_mask |= (1 << cand)
                                result.append(list(ALL_TRIPLES[cand]))
                                # Size unchanged, but we swapped. 
                                # This doesn't increase best_len, but might help future adds.
                                # We update best only if we find a larger set later.
                                # For now, we accept this state to explore.
                                # However, we should track the "current" best.
                                # If we just swapped, len is same. 
                                # We don't update best_result unless we grow.
                                # But we should keep the current state for further search.
                            else:
                                # Failed to add cand even after removal. 
                                # Restore? Or just keep the removal?
                                # Keeping the removal might be bad if it was useful.
                                # Let's restore if we couldn't add cand.
                                # Actually, let's just revert this specific attempt to avoid wasting time.
                                result.append(list(rem_triple))
                                selected_mask |= (1 << rem_idx)
                                # Try a different removal? 
                                # For simplicity, just give up on this cand for this iteration.

        # Strategy 2: Remove 1, Add 2 (Net +1)
        else:
            # Pick a random triple to remove
            if not result:
                continue
            pos = random.randrange(len(result))
            rem_triple = result[pos]
            rem_idx = TRIPLE_IDX[tuple(sorted(rem_triple))]
            
            # Tentatively remove
            new_mask = selected_mask & ~(1 << rem_idx)
            
            # Try to add two random triples
            added_any = False
            added_list = []
            
            for _ in range(2):
                cand = random.randrange(N)
                if not (new_mask >> cand) & 1:
                    if _can_add(new_mask, cand):
                        new_mask |= (1 << cand)
                        added_list.append(cand)
                        added_any = True
            
            if added_any and len(added_list) == 2:
                # Success: removed 1, added 2. Net gain +1.
                # Update result list
                result.pop(pos)
                for c in added_list:
                    result.append(list(ALL_TRIPLES[c]))
                selected_mask = new_mask
                best_result = result[:]
                best_mask = selected_mask
                best_len += 1

    return best_result, best_mask
=======
def _local_search(result, selected_mask, deadline, max_iters=10000):
    """
    Improve by:
    1. Trying to add a random triple.
    2. If blocked, try to remove one triple that is part of a blocking witness, then add the candidate.
    3. Try "remove 1, add 2" moves to increase size.
    Uses an in_set array for O(1) membership checks and removals.
    """
    # in_set[i] is True if triple i is in the current set
    in_set = [False] * N
    for i in range(len(result)):
        in_set[TRIPLE_IDX[tuple(sorted(result[i]))]] = True
        
    best_result = result[:]
    best_mask = selected_mask
    best_len = len(result)
    
    # List of currently selected triple indices for random removal
    selected_indices = [TRIPLE_IDX[tuple(sorted(t))] for t in result]
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Strategy 1: Add a random triple (50%)
        # Strategy 2: Swap (1-out, 1-in) (30%)
        # Strategy 3: Improve (1-out, 2-in) (20%)
        
        r = random.random()
        
        if r < 0.5:
            # Try to add a random triple
            cand = random.randrange(N)
            if not in_set[cand]:
                if _can_add(selected_mask, cand):
                    # Success
                    in_set[cand] = True
                    selected_mask |= (1 << cand)
                    selected_indices.append(cand)
                    result.append(list(ALL_TRIPLES[cand]))
                    best_result = result[:]
                    best_mask = selected_mask
                    best_len += 1
                else:
                    # Blocked. Try to unblock by removing a triple from a blocking witness.
                    # This is a swap attempt.
                    blocked_witnesses = [w for w in WITNESSES[cand] if (selected_mask & w) == w]
                    if blocked_witnesses:
                        w = random.choice(blocked_witnesses)
                        # Find bits in w
                        bits = []
                        tmp_w = w
                        while tmp_w:
                            lsb = tmp_w & -tmp_w
                            bits.append(lsb.bit_length() - 1)
                            tmp_w ^= lsb
                        
                        rem_idx = random.choice(bits)
                        
                        # Tentatively remove rem_idx
                        in_set[rem_idx] = False
                        selected_mask &= ~(1 << rem_idx)
                        # Remove from selected_indices (find and remove)
                        # Since selected_indices is a list, we can use remove which is O(n), but n is small (~200-300)
                        try:
                            selected_indices.remove(rem_idx)
                        except ValueError:
                            pass # Should not happen if in_set is consistent
                        # Remove from result list? We'll update result at the end if we keep the change.
                        # For now, we just track state.
                        
                        # Try to add cand
                        if _can_add(selected_mask, cand):
                            # Swap successful
                            in_set[cand] = True
                            selected_mask |= (1 << cand)
                            selected_indices.append(cand)
                            # Update result list: replace rem_idx with cand
                            # We need to find rem_idx in result and replace it
                            # Or just rebuild result if we accept this state?
                            # Rebuilding is O(n). Doing it every swap is expensive.
                            # Better: keep result in sync.
                            # Find rem_idx in result
                            rem_triple_list = list(ALL_TRIPLES[rem_idx])
                            pos = -1
                            for i, t in enumerate(result):
                                if t == rem_triple_list:
                                    pos = i
                                    break
                            if pos != -1:
                                result[pos] = list(ALL_TRIPLES[cand])
                            else:
                                # Fallback: rebuild
                                result = [list(ALL_TRIPLES[i]) for i in selected_indices]
                            
                            # Size unchanged, so best_len doesn't change.
                            # But we accept this state to explore further.
                            # We do NOT update best_result unless we grow.
                        else:
                            # Failed to add cand. Revert.
                            in_set[rem_idx] = True
                            selected_mask |= (1 << rem_idx)
                            selected_indices.append(rem_idx)
                            # Revert result list
                            rem_triple_list = list(ALL_TRIPLES[rem_idx])
                            cand_triple_list = list(ALL_TRIPLES[cand])
                            pos = -1
                            for i, t in enumerate(result):
                                if t == cand_triple_list:
                                    pos = i
                                    break
                            if pos != -1:
                                result[pos] = rem_triple_list
                            else:
                                result = [list(ALL_TRIPLES[i]) for i in selected_indices]

        elif r < 0.8:
            # Swap (1-out, 1-in) random
            if not selected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            
            in_set[rem_idx] = False
            selected_mask &= ~(1 << rem_idx)
            selected_indices.remove(rem_idx)
            
            cand = random.randrange(N)
            if not in_set[cand] and _can_add(selected_mask, cand):
                in_set[cand] = True
                selected_mask |= (1 << cand)
                selected_indices.append(cand)
                
                # Update result list
                rem_triple_list = list(ALL_TRIPLES[rem_idx])
                pos = -1
                for i, t in enumerate(result):
                    if t == rem_triple_list:
                        pos = i
                        break
                if pos != -1:
                    result[pos] = list(ALL_TRIPLES[cand])
                else:
                    result = [list(ALL_TRIPLES[i]) for i in selected_indices]
            else:
                # Revert
                in_set[rem_idx] = True
                selected_mask |= (1 << rem_idx)
                selected_indices.append(rem_idx)
                # Revert result list if we modified it? We didn't modify result in the failed branch.
                # Wait, in the failed branch of the previous strategy, we reverted.
                # Here, we didn't modify result yet. So just revert state.
                # Actually, we need to ensure result is consistent with in_set.
                # If we revert in_set, result is already consistent (we didn't change it).

        else:
            # Improve (1-out, 2-in)
            if not selected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            
            in_set[rem_idx] = False
            selected_mask &= ~(1 << rem_idx)
            selected_indices.remove(rem_idx)
            
            added_list = []
            # Try to add 2 random triples
            for _ in range(2):
                cand = random.randrange(N)
                if not in_set[cand] and _can_add(selected_mask, cand):
                    in_set[cand] = True
                    selected_mask |= (1 << cand)
                    selected_indices.append(cand)
                    added_list.append(cand)
            
            if len(added_list) == 2:
                # Success
                # Update result list
                rem_triple_list = list(ALL_TRIPLES[rem_idx])
                pos = -1
                for i, t in enumerate(result):
                    if t == rem_triple_list:
                        pos = i
                        break
                if pos != -1:
                    result[pos] = list(ALL_TRIPLES[added_list[0]])
                    result.append(list(ALL_TRIPLES[added_list[1]]))
                else:
                    result = [list(ALL_TRIPLES[i]) for i in selected_indices]
                
                best_result = result[:]
                best_mask = selected_mask
                best_len += 1
            else:
                # Revert
                in_set[rem_idx] = True
                selected_mask |= (1 << rem_idx)
                selected_indices.append(rem_idx)
                for c in added_list:
                    in_set[c] = False
                    selected_mask &= ~(1 << c)
                    selected_indices.remove(c)
                # Revert result list if we modified it?
                # We only modified result if we succeeded.
                # If we failed, we didn't modify result.
                # So result is consistent.

    return best_result, best_mask
>>>>>>> REPLACE

