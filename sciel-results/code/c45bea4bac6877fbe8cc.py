        # Pruning: if remaining candidates == slots left, we must take all
        slots_left = target_size - current_size
        rem_cands = n_cand - idx
        if rem_cands == slots_left:
            # We must include all remaining candidates
            # Check if adding all of them keeps the set valid
            # We can simulate adding them one by one
            
            temp_sumsets = [set(s) for s in sumsets]
            temp_chosen = chosen[:]
            
            for j in range(idx, n_cand):
                x = candidates[j]
                # Update temp_sumsets with x
                for c in range(len(temp_sumsets) - 1, 0, -1):
                    if temp_sumsets[c-1]:
                        for s in temp_sumsets[c-1]:
                            temp_sumsets[c].add(s + x)
                
                # Check validity
                if not _check_partial_validity(temp_sumsets):
                    return False # Dead end
            
            # If we get here, all remaining were added and valid
            # But wait, we need to update `chosen` and `sumsets` to reflect this?
            # Actually, if this path is valid, we have found a solution.
            # We just need to return True and ensure `chosen` is correct.
            # But `chosen` is not updated in the loop above.
            # So we should just return True, and the caller will see it as success.
            # However, the `chosen` list in the caller's scope won't have these elements.
            # This is a problem if we rely on `chosen` being correct after `backtrack` returns True.
            
            # Better: Just return True. The parent will return True.
            # The top-level caller will then use `chosen`.
            # But `chosen` is missing the last few elements.
            
            # Fix: Append them to chosen before returning True.
            for j in range(idx, n_cand):
                chosen.append(candidates[j])
            return True

