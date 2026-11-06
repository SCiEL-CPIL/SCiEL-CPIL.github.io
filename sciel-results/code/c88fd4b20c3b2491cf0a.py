"""Rewritten program using a local search approach starting from Odds + Powers of 2."""
import math

# EVOLVE-BLOCK-START
def construct():
    N = 256
    
    def conflicts(a, b):
        if a == b:
            return False
        return (a * b) % (a + b) == 0

    odds = [i for i in range(1, N + 1) if i % 2 != 0]
    evens = [i for i in range(2, N + 1, 2)]
    powers_of_2 = [1 << i for i in range(9) if (1 << i) <= N]
    
    S = set(odds)
    S.update(powers_of_2)
    
    # Map odds to indices for bitmask
    odd_to_idx = {o: i for i, o in enumerate(odds)}
    num_odds = len(odds)
    
    # Precompute conflict masks for evens against odds
    # mask[e] has bit i set if e conflicts with odds[i]
    mask_conflicts_odds = {}
    for e in evens:
        mask = 0
        for o in odds:
            if conflicts(e, o):
                mask |= (1 << odd_to_idx[o])
        mask_conflicts_odds[e] = mask
        
    # Greedy addition of evens
    changed = True
    while changed:
        changed = False
        for e in evens:
            if e in S:
                continue
            safe = True
            for x in S:
                if conflicts(e, x):
                    safe = False
                    break
            if safe:
                S.add(e)
                changed = True

    # Local Search
    # We will try to remove sets of odds of size 1 and 2 to add more evens.
    
    # Precompute list of evens not in S for faster iteration? 
    # S changes, so we just check `if e in S`.
    
    def try_remove_and_add(removed_odds):
        """
        Attempts to remove the specified set of odds and add all possible safe evens.
        Returns True if the net size increased, and modifies S if so.
        """
        if not removed_odds:
            return False
            
        # Check if removed odds are actually in S
        for o in removed_odds:
            if o not in S:
                return False
                
        # Calculate mask of removed odds
        mask_removed = 0
        for o in removed_odds:
            mask_removed |= (1 << odd_to_idx[o])
            
        # Temporarily remove
        for o in removed_odds:
            S.remove(o)
            
        added_evens = []
        
        # Find candidates
        # An even e is a candidate if all its odd-conflicts are in removed_odds
        # AND it does not conflict with any other element in S (which is S \ removed_odds)
        
        for e in evens:
            if e in S:
                continue
                
            # Check odd conflicts
            if mask_conflicts_odds[e] & ~mask_removed:
                continue # Conflicts with an odd not in removed set
                
            # Check conflicts with other elements in S
            safe = True
            for x in S:
                if x == e: continue
                if conflicts(e, x):
                    safe = False
                    break
                    
            if safe:
                S.add(e)
                added_evens.append(e)
                
        # Evaluate gain
        removed_count = len(removed_odds)
        added_count = len(added_evens)
        
        if added_count > removed_count:
            # Keep the change
            return True
        else:
            # Revert
            for e in added_evens:
                S.remove(e)
            for o in removed_odds:
                S.add(o)
            return False

    improved = True
    while improved:
        improved = False
        
        # Try removing 1 odd
        for o in odds:
            if o in S:
                if try_remove_and_add([o]):
                    improved = True
                    # Restart loop? Or continue?
                    # If we add evens, the set of safe evens changes.
                    # Restarting is safer to find new opportunities.
                    break
        
        if improved:
            continue
            
        # Try removing 2 odds
        # To speed up, we can iterate pairs.
        # 128 odds -> 8128 pairs.
        # Each try_remove_and_add is O(N_evens * |S|) in worst case.
        # 8128 * 128 * 150 ~ 150M ops. Might be slow in Python.
        
        # Optimization: Only try pairs if we have time?
        # Or limit pairs.
        
        # Let's try a heuristic: Only try pairs if single removals failed.
        # And maybe only try a subset of pairs?
        # Or just try all, but break early if found.
        
        found_pair = False
        # To avoid O(N^2) full scan every time, we can just do one pass.
        # If we find an improvement, we break and restart.
        
        for i in range(num_odds):
            o1 = odds[i]
            if o1 not in S:
                continue
            for j in range(i + 1, num_odds):
                o2 = odds[j]
                if o2 not in S:
                    continue
                
                # Heuristic: Check if removing o1 and o2 allows adding evens
                # Quick check: Are there evens whose conflict mask is subset of {o1, o2}?
                # We can precompute this? No, S changes.
                # But the odd conflict mask is static.
                # So we can precompute for each pair of odds, the list of evens that ONLY conflict with that pair (or subset).
                # This is expensive to precompute for all pairs.
                
                # Let's just call try_remove_and_add.
                # To make it faster, we can check a few candidates first.
                
                if try_remove_and_add([o1, o2]):
                    improved = True
                    found_pair = True
                    break
            if found_pair:
                break
                
    return list(S)
# EVOLVE-BLOCK-END
