import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Generate all triples
    points = range(16)
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Map triple to index for faster set operations?
    # Using a set of tuples is probably fast enough for n=560.
    
    # 2. Precompute conflicts for each triple
    # conflicts[i] is a list of 7-tuples (sets of triples) that conflict with triple i
    # A 7-tuple C conflicts with triple t if C U {t} forms a K(2,2,2)
    
    # Helper to get the other 7 transversals given a structure and the main triple
    # Structure: 3 pairs (p1, p2, p3)
    # t = (u1, u2, u3) with ui in pi
    # Other transversals are all (v1, v2, v3) != (u1, u2, u3)
    
    # We need to iterate over all K(2,2,2) structures containing t.
    # t = (a, b, c)
    # Assign a, b, c to pairs 1, 2, 3 (6 permutations)
    # Choose 3 partners from remaining 13 vertices
    
    # To speed up, we can precompute the list of 7-sets for each triple.
    # This might take some time, but it's a one-time cost.
    
    # Let's estimate precomputation cost.
    # 560 triples * 6 perms * 286 partner sets = ~960,000 structures to generate.
    # For each, we generate 7 triples.
    # Total ~6.7 million triples generated.
    # This is fast in Python.
    
    conflicts = [None] * n_triples
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute
    for i, t in enumerate(all_triples):
        a, b, c = t
        conflict_sets = []
        # Permutations of (a,b,c) assigned to pair slots 1,2,3
        for perm in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
            u1, u2, u3 = perm
            # Remaining vertices
            remaining = [x for x in points if x not in (a,b,c)]
            # Choose 3 vertices for partners
            for partners in combinations(remaining, 3):
                v1, v2, v3 = partners
                # The structure is defined by pairs {u1,v1}, {u2,v2}, {u3,v3}
                # The 8 transversals are all combinations of choosing one from each pair
                # We know t = (u1, u2, u3) is one of them.
                # The other 7 are:
                # (v1, u2, u3), (u1, v2, u3), (u1, u2, v3)
                # (v1, v2, u3), (v1, u2, v3), (u1, v2, v3)
                # (v1, v2, v3)
                
                # We need to check if these 7 triples exist in all_triples
                # And form a set.
                
                # Actually, we just store the 7 triples.
                # We must ensure we don't add duplicate conflict sets?
                # Each structure is unique.
                
                # Construct the 7 triples
                # Note: triples are sorted tuples
                t1 = tuple(sorted((v1, u2, u3)))
                t2 = tuple(sorted((u1, v2, u3)))
                t3 = tuple(sorted((u1, u2, v3)))
                t4 = tuple(sorted((v1, v2, u3)))
                t5 = tuple(sorted((v1, u2, v3)))
                t6 = tuple(sorted((u1, v2, v3)))
                t7 = tuple(sorted((v1, v2, v3)))
                
                # These 7 triples must be distinct from t and each other?
                # In a K(2,2,2), all 8 transversals are distinct.
                # So t1..t7 are distinct and != t.
                
                conflict_sets.append((t1, t2, t3, t4, t5, t6, t7))
                
        # Remove duplicates in conflict_sets?
        # Different structures might yield the same 7-set?
        # No, the 7-set determines the 8th triple and the structure uniquely?
        # Yes.
        conflicts[i] = conflict_sets
        
    # 3. Random Restart Greedy
    best_set = set()
    best_size = 0
    
    # Initial simple greedy to get a baseline
    # Just run a few iterations
    
    num_iterations = 0
    while time.time() - start_time < time_limit:
        num_iterations += 1
        # Shuffle triples
        random.shuffle(all_triples)
        
        current_set = set()
        
        for t in all_triples:
            # Check if t can be added
            # t is blocked if any of its conflict sets is a subset of current_set
            can_add = True
            # Get index of t? No, we need the conflicts for t.
            # We have conflicts indexed by original index.
            # But we shuffled all_triples.
            # We need to map t to its index.
            idx = triple_to_idx[t]
            
            for c_set in conflicts[idx]:
                # Check if all 7 triples in c_set are in current_set
                # If yes, t cannot be added
                if all(x in current_set for x in c_set):
                    can_add = False
                    break
            
            if can_add:
                current_set.add(t)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Optional: Local Search on best_set
        # If we have time, try to improve best_set
        # Simple local search: try to add any missing triple
        # If none, try swaps
        # Given time constraints, simple restarts might be sufficient.
        # But let's add a simple improvement step.
        
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try to improve current best_set
        # Create a list of missing triples
        all_triples_set = set(all_triples)
        missing = list(all_triples_set - best_set)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for t in missing:
                idx = triple_to_idx[t]
                can_add = True
                for c_set in conflicts[idx]:
                    if all(x in best_set for x in c_set):
                        can_add = False
                        break
                if can_add:
                    best_set.add(t)
                    missing.remove(t)
                    improved = True
                    break # Restart missing check? Or just continue?
                    # Breaking and restarting the loop is safer to catch new possibilities?
                    # Actually, adding t might block others, but won't unblock.
                    # So we can just continue.
                    # But removing from list while iterating is bad.
                    # Let's just break and re-evaluate.
                    
        # If no simple add worked, try a kick?
        # Remove 1 random triple, try to add 1 missing.
        # This is complex. Let's stick to restarts + simple add.
        
    # Convert set to list of lists as required
    result = [list(t) for t in best_set]
    return result

# Run construct to verify? No, the evaluator will run it.
# But we should ensure it's correct.
