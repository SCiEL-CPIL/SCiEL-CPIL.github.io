"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 180
    # 1. Find all forbidden triples (a,b,c) distinct in [1,N] with 1/a = 1/b + 1/c
    # c = a*b/(b-a). b > a.
    forbidden = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    forbidden.add(frozenset((a, b, c)))
    
    # 2. Parity-based partition
    # All odd numbers are safe from each other.
    odds = [x for x in range(1, N + 1) if x % 2 != 0]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    
    # 3. Determine candidate evens
    # An even number e is a candidate if it does NOT form a triple with two distinct odds.
    # If it does, we must exclude it because we keep all odds.
    odd_set = set(odds)
    candidate_evens = []
    
    # Map even -> list of triples it's in
    # Or just check directly
    # To speed up, precompute for each even e, if there is a triple {e, o1, o2}
    
    # Let's build a set of forbidden pairs of evens (due to odd conflict) and forbidden triples of evens
    # pair_conflict[u, v] = True if exists o in odds s.t. {u, v, o} in forbidden
    # triple_conflict[u, v, w] = True if {u, v, w} in forbidden
    
    # First, filter candidate evens
    # For each even e, check if any triple {e, x, y} exists where x,y are odd.
    # If yes, e is NOT a candidate (we keep all odds, so e must be excluded).
    
    # Efficient check:
    # Iterate through forbidden triples.
    # If a triple has exactly one even and two odds, that even is "poisoned".
    # If a triple has two evens and one odd, that pair of evens is "conflicting".
    # If a triple has three evens, that triple is "conflicting".
    
    poisoned_evens = set()
    even_pair_conflicts = set() # frozenset({u, v})
    even_triple_conflicts = set() # frozenset({u, v, w})
    
    for tri in forbidden:
        evens_in_tri = [x for x in tri if x % 2 == 0]
        odds_in_tri = [x for x in tri if x % 2 != 0]
        
        if len(evens_in_tri) == 1:
            # One even, two odds. The even is poisoned.
            poisoned_evens.add(evens_in_tri[0])
        elif len(evens_in_tri) == 2:
            # Two evens, one odd. The pair of evens conflicts.
            even_pair_conflicts.add(frozenset(evens_in_tri))
        elif len(evens_in_tri) == 3:
            # Three evens. The triple conflicts.
            even_triple_conflicts.add(frozenset(evens_in_tri))
    
    candidate_evens = [e for e in evens if e not in poisoned_evens]
    
    # 4. Solve the subproblem on candidate_evens
    # We want to select max subset of candidate_evens such that:
    # - No pair in even_pair_conflicts is both selected.
    # - No triple in even_triple_conflicts is all selected.
    
    # Greedy + Local Search
    
    def solve_evens(candidates):
        if not candidates:
            return set()
        
        # Precompute adjacency for pairs
        pair_adj = {e: set() for e in candidates}
        for pair in even_pair_conflicts:
            u, v = list(pair)
            if u in pair_adj and v in pair_adj:
                pair_adj[u].add(v)
                pair_adj[v].add(u)
        
        # Precompute triples involving candidates
        # triple_adj[e] = set of frozensets of other 2 candidates that form a conflict triple with e
        # Or just keep the list of triples for checking
        
        # Greedy initialization
        # Sort by degree (pair conflicts) ascending
        sorted_cands = sorted(candidates, key=lambda x: len(pair_adj[x]))
        
        current_set = set()
        # trip_count for triples? 
        # We need to check:
        # 1. Pair conflict: if u, v in set and (u,v) in pair_conflicts -> invalid
        # 2. Triple conflict: if u, v, w in set and {u,v,w} in triple_conflicts -> invalid
        
        # For efficiency, maintain:
        # - current_set
        # - For each element in current_set, its "active" pair conflicts?
        # Actually, just checking validity on addition is O(degree + triple_degree).
        
        # Let's use a simple greedy first
        for e in sorted_cands:
            can_add = True
            # Check pair conflicts with existing elements
            for other in current_set:
                if other in pair_adj[e]:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Check triple conflicts
            # If we add e, we form a triple {e, x, y} if x,y in current_set and {e,x,y} in even_triple_conflicts
            # We need to check if there are x,y in current_set such that {e,x,y} is a conflict.
            # Precompute for each e, the set of pairs (x,y) in candidates that conflict with e.
            # triple_pairs[e] = set of frozenset({x,y})
            
            # Let's precompute this outside
            # But inside this function, we can just iterate
            # Optimization: 
            # For each e, find all x in current_set.
            # For each x, find all y in current_set (y != x).
            # Check if {e,x,y} in even_triple_conflicts.
            # This is O(|current_set|^2) which might be slow if |current_set| is large (up to 90).
            # 90^2 = 8100. 90 * 8100 = 729,000 ops. Acceptable.
            
            # Better: Precompute triple_neighbors[e] = set of frozensets of other 2 elements
            # Then check if any neighbor is subset of current_set.
            
            # Let's just do the naive check for now, it's small N.
            
            # Actually, let's precompute triple_adjacency for candidates
            # triple_adj[e] = list of frozenset({x,y})
            
            # We'll do a local search to improve.
            
            # For now, add if valid
            # Check triples
            valid_triple = True
            # To make this faster, we can store for each e, the set of "forbidden pairs"
            # forbidden_pairs_for_e = set of frozenset({x,y}) such that {e,x,y} in even_triple_conflicts
            
            # Let's assume we have a helper function is_valid_addition(e, current_set)
            
            # For speed in the loop, let's just try to add and check.
            # We'll use a global precomputed structure.
            
            # Let's refine the data structures before the loop
            # We need triple_adj for candidates
            # triple_adj[e] = set of frozenset({x,y})
            
            # We'll build this in the main function and pass it? 
            # Or build it here.
            
            # Let's just check validity by iterating current_set pairs.
            # Since N=180, evens=90, candidates <= 90.
            # Greedy pass is fast enough.
            
            # Check triple validity
            # We need to know if there exist x, y in current_set such that {e, x, y} is a conflict.
            # We can precompute for each e, a set of "forbidden pairs" from the candidates.
            # Let's do this precomputation once.
            
            pass # Placeholder for logic restructuring
            
        # Let's implement a proper local search on the evens
        
        # Precompute triple constraints for candidates
        # triple_conf[e] = set of frozenset({x,y})
        triple_conf = {e: set() for e in candidates}
        for tri in even_triple_conflicts:
            elements = list(tri)
            for i in range(3):
                e = elements[i]
                if e in triple_conf:
                    other_two = frozenset(elements[:i] + elements[i+1:])
                    triple_conf[e].add(other_two)
        
        def is_valid(e, s):
            # Check pair conflicts
            for other in s:
                if other in pair_adj[e]:
                    return False
            # Check triple conflicts
            # Is there any pair in s that is in triple_conf[e]?
            # Iterate pairs in s? O(|s|^2)
            # Or iterate triple_conf[e] and check if subset of s?
            # triple_conf[e] size is small.
            for pair in triple_conf[e]:
                if pair.issubset(s):
                    return False
            return True

        # Greedy
        current_set = set()
        for e in sorted_cands:
            if is_valid(e, current_set):
                current_set.add(e)
        
        # Local Search
        # Try to add excluded, or swap
        excluded = set(candidates) - current_set
        
        improved = True
        while improved:
            improved = False
            
            # Try additions
            added = False
            for e in list(excluded):
                if is_valid(e, current_set):
                    current_set.add(e)
                    excluded.discard(e)
                    improved = True
                    added = True
                    break # Restart to catch new opportunities
            
            if added:
                continue
            
            # Try swaps
            # Remove x, add y
            # We want to maintain size or increase? 
            # Swapping keeps size same. We do swaps to enable future additions.
            # So we do a swap if it allows at least one addition afterwards.
            # Or just do a swap if it doesn't reduce size (it doesn't) and we hope for the best?
            # Better: Only swap if it leads to a valid state (it does) and we can find an addition later.
            # Simple heuristic: Try all swaps, keep the one that allows the most immediate additions?
            # Too expensive.
            # Just try random swaps or first valid swap?
            # Let's try: for each x in current_set, for each y in excluded, 
            # if (current_set - {x} + {y}) is valid (it is, since we removed x), 
            # check if any z in excluded - {y} can be added to (current_set - {x} + {y}).
            # If yes, perform the swap and the addition.
            
            swap_made = False
            for x in list(current_set):
                # Temporarily remove x
                new_set = current_set - {x}
                new_excluded = excluded | {x}
                
                # Try to add y from new_excluded
                # We want to find a y such that adding y allows adding z.
                # Or just add y if it's "better"?
                # Let's just try to add any y, then try to add z.
                # To be efficient, we can check if the degree of new_set is lower?
                
                # Let's try a simpler swap: 
                # If we remove x, can we add y?
                # Yes, always, because we removed a constraint.
                # But does it help?
                # Let's just try to add y, then check if we can add z.
                
                # We'll iterate y
                for y in list(new_excluded):
                    if y == x: continue
                    # Add y to new_set
                    test_set = new_set | {y}
                    test_excluded = new_excluded - {y}
                    
                    # Check if we can add any z from test_excluded
                    can_add_more = False
                    for z in test_excluded:
                        if is_valid(z, test_set):
                            can_add_more = True
                            # Perform the operation: swap x->y and add z
                            current_set = test_set
                            current_set.add(z)
                            excluded = test_excluded - {z}
                            improved = True
                            swap_made = True
                            break
                    if swap_made:
                        break
                if swap_made:
                    break
            
        return current_set

    best_even_set = solve_evens(candidate_evens)
    
    final_set = set(odds)
    final_set.update(best_even_set)
    
    return sorted(final_set)
# EVOLVE-BLOCK-END
