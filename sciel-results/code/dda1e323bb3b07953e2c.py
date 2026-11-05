"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    N = 128
    all_numbers = list(range(1, N + 1))
    
    # Precompute LCMs for speed
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_table[(a, b)] = lcm(a, b)
            
    # Precompute all bad triples
    # A triple {a, b, c} is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    triple_indices = {} # map triple to index? or just list of sets
    # For efficiency, we store triples as tuples of sorted indices
    # We also need to quickly check if a triple is "active" (all in S)
    
    # Let's identify all bad triples first
    for a, b, c in combinations(all_numbers, 3):
        if lcm_table[(a, b)] == lcm_table[(a, c)] == lcm_table[(b, c)]:
            bad_triples.append((a, b, c))
            
    if not bad_triples:
        return all_numbers
        
    # Map number to list of triple indices it belongs to
    num_to_triples = {x: [] for x in all_numbers}
    for i, t in enumerate(bad_triples):
        for x in t:
            num_to_triples[x].append(i)
            
    best_set = set()
    best_size = 0
    
    # Number of restarts
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # 1. Start with all numbers
        S = set(all_numbers)
        
        # 2. Remove elements to break all bad triples
        # We maintain a count of how many bad triples are "active" (all 3 in S)
        # Initially all are active.
        # We remove elements one by one.
        
        # To speed up, we can track for each triple if it's still "alive"
        # A triple is alive if all its elements are in S.
        
        # Heuristic: Always remove the element involved in the most "alive" triples.
        
        # Efficiently tracking "alive" triples is tricky with set removals.
        # Let's use a simpler loop. Since N=128 and bad_triples count is likely < 1000, 
        # we can just re-scan or use a counter.
        
        # Let's count active triples per number.
        # active_count[x] = number of bad triples containing x where all 3 elements are in S.
        
        active_count = {x: len(num_to_triples[x]) for x in all_numbers}
        
        # We need to update active_count when we remove an element.
        # If we remove x, all triples containing x become inactive.
        # For each other element y in those triples, its active_count decreases.
        
        # This is O(E) per removal. E ~ 3 * |bad_triples|.
        # |bad_triples| is small.
        
        # Let's store triples as lists of 3 numbers.
        # And for each number, list of triples.
        
        # To check if a triple is active, we need to know if all 3 are in S.
        # We can just check membership in S.
        
        # Algorithm:
        # While there exists a bad triple t such that t subset S:
        #   Find the x in S that is in the most "active" bad triples.
        #   Remove x from S.
        #   Update active counts.
        
        # To avoid O(N * T) scan for "most active", we can just pick a random active triple 
        # and remove the "most connected" element from it.
        
        # Let's find all active triples initially.
        # Actually, we can just iterate:
        
        # Create a list of active triples
        # We'll use a set of triple indices that are currently active.
        active_triples = set(range(len(bad_triples)))
        
        # To quickly find an active triple, we can just pick a random one from active_triples.
        # But we need to ensure it's actually active (all in S).
        # Since we only remove elements, if it was active, it stays active or becomes inactive.
        # Wait, if we remove an element, the triples containing it become inactive.
        # So we can maintain active_triples correctly.
        
        # Initialize active_triples
        # All are active initially.
        
        # Counters: how many active triples does each number participate in?
        # This helps choose which number to remove.
        deg = {x: len(num_to_triples[x]) for x in all_numbers}
        
        while active_triples:
            # Pick a random active triple
            t_idx = random.choice(list(active_triples))
            a, b, c = bad_triples[t_idx]
            
            # Check if it's still active (should be, but just in case)
            if a in S and b in S and c in S:
                # Choose the element to remove: the one with highest degree in active triples
                # We need the current degree of a, b, c in active triples.
                # deg[x] is the count of active triples containing x.
                
                # Find max degree among a, b, c
                # Note: deg is updated as we remove elements.
                # When we remove x, we decrement deg for all neighbors in active triples.
                
                # Let's just pick the one with max deg among the three.
                # Ties broken randomly.
                candidates = [(deg[a], a), (deg[b], b), (deg[c], c)]
                candidates.sort(key=lambda x: -x[0])
                max_deg = candidates[0][0]
                # Random among max
                top_candidates = [x for d, x in candidates if d == max_deg]
                to_remove = random.choice(top_candidates)
                
                # Remove to_remove from S
                S.remove(to_remove)
                
                # Update active_triples and deg
                # All triples containing to_remove become inactive
                for t_idx2 in num_to_triples[to_remove]:
                    if t_idx2 in active_triples:
                        active_triples.remove(t_idx2)
                        # Decrement deg for the other two elements in this triple
                        t = bad_triples[t_idx2]
                        for y in t:
                            if y != to_remove:
                                deg[y] -= 1
            else:
                # This triple is no longer active? 
                # This shouldn't happen if we remove from active_triples correctly.
                # But if we picked a stale index?
                # We are picking from active_triples, so it should be active.
                # Unless my logic for removing from active_triples missed something.
                # We remove t_idx2 from active_triples when we remove an element.
                # So any t_idx in active_triples should have all elements in S.
                # So this branch should never trigger.
                pass
        
        # Now S is valid.
        
        # 3. Try to add elements back
        # Candidates are all_numbers not in S
        removed = [x for x in all_numbers if x not in S]
        random.shuffle(removed)
        
        for x in removed:
            if time.time() - start_time > timeout:
                break
            # Check if adding x creates any bad triple
            # A bad triple involving x must be of the form {x, u, v} where {u,v} in S
            # We can precompute for each x, the list of pairs {u,v} such that {x,u,v} is a bad triple.
            # Let's build this map now? No, do it on the fly or precompute once.
            
            # Precompute once outside the loop? 
            # x_to_pairs[x] = list of (u, v) such that {x,u,v} is bad.
            # This is static.
            
            # Let's assume we have this map.
            # For now, let's just check all pairs in S? Too slow.
            
            # We'll precompute x_to_bad_pairs in the main scope? 
            # It depends on bad_triples.
            
            # Let's build it once before the restart loop.
            # But I didn't. Let's do a quick check.
            # For a given x, and current S, 
            # Check if there exist u, v in S such that {x, u, v} is a bad triple.
            
            # We can iterate through num_to_triples[x]
            # For each triple index t_idx in num_to_triples[x]:
            #   Get the other two elements u, v.
            #   If u in S and v in S, then adding x creates a conflict.
            
            can_add = True
            for t_idx in num_to_triples[x]:
                t = bad_triples[t_idx]
                # Find the other two
                u = v = None
                for y in t:
                    if y != x:
                        if u is None:
                            u = y
                        else:
                            v = y
                if u in S and v in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return list(best_set) if best_set else [1]

# EVOLVE-BLOCK-END
