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
    
    # Initial set: All odds and all powers of 2
    # Powers of 2 are pairwise safe and safe with all odds.
    S = set(odds)
    S.update(powers_of_2)
    
    # Precompute conflicts for all evens against all odds
    # conflict_map[e] = set of odds that conflict with e
    conflict_map = {}
    for e in evens:
        conflicts_with_odds = set()
        for o in odds:
            if conflicts(e, o):
                conflicts_with_odds.add(o)
        conflict_map[e] = conflicts_with_odds
        
    # Identify evens that are "safe" with all odds (conflict set is empty)
    # These can be added freely (subject to even-even conflicts)
    # We already added powers of 2. 
    # Let's add other safe evens greedily.
    
    # Sort evens not in S by some heuristic?
    # Let's just iterate and add if safe with current S.
    # But we need to be careful about even-even conflicts.
    
    # Let's do a pass to add any even that has no conflicts with current S.
    changed = True
    while changed:
        changed = False
        for e in evens:
            if e in S:
                continue
            # Check if e conflicts with any element in S
            safe = True
            for x in S:
                if conflicts(e, x):
                    safe = False
                    break
            if safe:
                S.add(e)
                changed = True
                
    # Now, try local search: 
    # For each odd o, check if removing o allows adding more evens than 1.
    # We track the current size.
    current_size = len(S)
    
    # We need to efficiently check what evens become available if we remove o.
    # An even e becomes available if:
    # 1. e is not in S.
    # 2. e does not conflict with any element in S \ {o}.
    # 3. e conflicts with o (otherwise it would have been added in the previous phase, 
    #    unless it conflicts with another element in S).
    # Actually, if e conflicts with o AND another element x in S, removing o doesn't help.
    # So we look for e such that conflict_map[e] is a subset of {o} 
    # AND e does not conflict with any even in S.
    
    # Let's precompute even-even conflicts? 
    # It's O(E^2) which is 128^2 = 16384. Fine.
    # But S changes.
    
    # Simplified Local Search:
    # Iterate through odds. For each, simulate removal.
    # If gain > 0, commit.
    
    # To speed up, we can just try it.
    # Max 128 odds. For each, check ~128 evens. 128*128 = 16384 checks.
    # Each check involves iterating S (size ~130). 16384 * 130 ~ 2M ops. Fast.
    
    for _ in range(10): # A few iterations of local search
        improved = False
        for o in odds:
            if o not in S:
                continue
            
            # Tentatively remove o
            S.remove(o)
            
            # Try to add evens
            added_count = 0
            added_evens = []
            
            for e in evens:
                if e in S:
                    continue
                
                # Check if e is now safe
                # It must not conflict with any x in S
                safe = True
                for x in S:
                    if conflicts(e, x):
                        safe = False
                        break
                
                if safe:
                    S.add(e)
                    added_evens.append(e)
                    added_count += 1
            
            # Net change: -1 (removed o) + added_count
            if added_count > 1:
                # We gained size. Keep the change.
                improved = True
            else:
                # We didn't gain size (or lost). Revert.
                S.add(o)
                for e in added_evens:
                    S.remove(e)
            
            if not improved:
                # If no single odd removal improves, try swapping 2 odds? 
                # Or just break if no improvement in this pass.
                pass
        
        if not improved:
            break
            
    return list(S)
# EVOLVE-BLOCK-END
