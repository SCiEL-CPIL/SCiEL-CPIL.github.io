"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    def is_sidon(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                s = selected[i] + selected[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                sums.add(selected[i] + selected[j])
        for x in range(1, N + 1):
            if x in selected:
                continue
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                return False
        return True

    def greedy_conflict_max():
        selected = []
        sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_block_count = -1
            
            for x in remaining:
                # Check if x can be added to selected
                added_sums = [x + a for a in selected] + [2 * x]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if sums.intersection(added_sums):
                    continue
                
                # If x can be added, how many other elements in 'remaining' would be blocked?
                # An element y is blocked if (y+a in new_sums) or (2y in new_sums) for some a in selected+{x}
                # new_sums = sums U added_sums
                new_sums = sums.union(added_sums)
                
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y is blocked by new_sums
                    blocked = False
                    # y is blocked if exists a in selected+{x} such that y+a in new_sums
                    # Note: a can be x or any a in selected
                    # If a=x, y+x in new_sums? y+x is in added_sums? No, added_sums has x+a for a in selected.
                    # Wait, the condition for y to be blocked by the set S' = S U {x} is:
                    # There exists a in S' such that y+a in Sum(S').
                    # S' = S U {x}.
                    # Sum(S') = Sum(S) U {x+a for a in S} U {2x}.
                    
                    # So we check if any y+a is in new_sums for a in S U {x}.
                    # a can be x: y+x in new_sums?
                    # a can be a in S: y+a in new_sums?
                    
                    if (y + x) in new_sums:
                        blocked = True
                    else:
                        for a in selected:
                            if (y + a) in new_sums:
                                blocked = True
                                break
                    if not blocked and (2 * y) in new_sums:
                        blocked = True
                        
                    if blocked:
                        block_count += 1
                
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_x = x
                    best_added = added_sums
            
            if best_x is None:
                # Should not happen if we start with empty set, but safety
                break
                
            selected.append(best_x)
            sums.update(best_added)
            remaining.remove(best_x)
            
            # Optimization: Remove any remaining elements that are now blocked?
            # No, we just pick the next one from remaining.
            # But we can prune remaining to only those that are NOT yet blocked?
            # If y is already blocked by current S, it doesn't matter if we pick x or not, y is out.
            # But we are constructing S. We want S to be maximal.
            # If y is blocked by S, it cannot be in S.
            # So we can remove y from 'remaining' candidates for the next step?
            # Yes! If y is blocked by current S, it will never be added to S.
            # So we can remove it from 'remaining' to speed up the search.
            
            new_remaining = set()
            for y in remaining:
                # Check if y is blocked by current S (which now includes best_x)
                blocked = False
                if (y + best_x) in sums: # best_x is in S, so y+best_x is a sum to check?
                    # Wait, the condition is y+a in Sum(S).
                    # If a=best_x, we check y+best_x in Sum(S).
                    # Sum(S) contains best_x + a for a in S_old.
                    # Does Sum(S) contain y+best_x?
                    # Not necessarily.
                    # Let's just use the full check.
                    pass
                
                # Efficient check:
                # y is blocked if exists a in S such that y+a in Sum(S) OR 2y in Sum(S).
                # S is 'selected'. Sum(S) is 'sums'.
                
                if (2 * y) in sums:
                    blocked = True
                else:
                    for a in selected:
                        if (y + a) in sums:
                            blocked = True
                            break
                
                if not blocked:
                    new_remaining.add(y)
            
            remaining = new_remaining
            
        return selected

    # Try the conflict-maximizing greedy
    best = greedy_conflict_max()
    
    if not is_maximal(best):
        # Fallback to random if it fails (shouldn't, but just in case)
        import random
        random.seed(42)
        for seed in range(10):
            random.seed(seed)
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            selected = []
            sums = set()
            for x in candidates:
                added = [x + a for a in selected] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    selected.append(x)
                    sums.update(added)
            if is_maximal(selected):
                best = selected
                break

    return best
# EVOLVE-BLOCK-END
