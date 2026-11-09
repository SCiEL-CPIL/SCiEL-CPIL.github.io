"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s kill

    n = 16
    points = list(range(n))

    best_alpha = n
    best_triples = []

    # Exact independence number calculation for N=16
    # We use bitmask enumeration. Since 2^16 = 65536, this is fast enough for many trials.
    # We optimize by checking subsets in descending order of size and stopping at the first valid one.
    # However, generating combinations for each size is expensive.
    # Alternative: Iterate all masks, compute popcount, track max valid.
    # Even better: Precompute list of all masks sorted by popcount descending? 
    # Or just iterate masks and keep track of the best.
    # Let's try the "iterate all masks" approach with a fast check.
    
    # To make it faster, we can precompute the list of triple masks.
    # A mask 'm' is independent if for all t in triple_masks, (m & t) != t.
    
    # We can optimize the check by noting that if we have a set of triples, 
    # we can just iterate through them.
    
    def get_alpha(triples):
        if not triples:
            return n
        
        t_masks = []
        for a, b, c in triples:
            t_masks.append((1 << a) | (1 << b) | (1 << c))
        
        # Sort t_masks by number of bits? They all have 3 bits.
        # We want to find the max popcount of a mask m such that no t is a subset of m.
        
        # Iterate masks from (1<<16)-1 down to 0 is not sorted by popcount.
        # Let's iterate by size from n-1 down to 0.
        # C(16, 15) = 16
        # C(16, 14) = 120
        # C(16, 13) = 560
        # C(16, 12) = 1820
        # C(16, 11) = 4368
        # C(16, 10) = 8008
        # C(16, 9) = 11440
        # C(16, 8) = 12870
        # Total is ~40k combinations. This is very fast.
        
        for size in range(n - 1, -1, -1):
            # Generate all combinations of 'size' points
            for comb in itertools.combinations(range(n), size):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                
                # Check if independent
                independent = True
                for t in t_masks:
                    if (mask & t) == t:
                        independent = False
                        break
                
                if independent:
                    return size
        return 0

    # Greedy construction with randomization
    # We maintain a set of used pairs to ensure linearity (no pair appears in more than one triple).
    # This is a 2-(16, 3, 1) packing design.
    
    def try_random_greedy():
        # Randomize the order of all possible triples
        all_triples = list(itertools.combinations(range(n), 3))
        random.shuffle(all_triples)
        
        used_pairs = set()
        triples = []
        
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            triples.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples

    # Local search / Hill Climbing
    # Move: Remove a triple, then try to add a different triple.
    # Or: Swap a point in a triple with a point from another triple? 
    # Linearity constraint makes swaps tricky.
    # Let's stick to "Remove and Re-add" or "Add if possible".
    
    def local_search(initial_triples):
        triples = list(initial_triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
        
        # Calculate current alpha
        current_alpha = get_alpha(triples)
        
        # Try to improve by removing triples that "cover" many points in the independent set?
        # Or just random restarts are better for this small N?
        
        # Let's try a simple swap:
        # Pick a random triple T1. Remove it.
        # Try to add a random triple T2 that is valid.
        # If T2 is valid and adding it doesn't increase alpha (or decreases it), keep it.
        # But checking alpha is expensive.
        
        # Alternative: Just do many random greedy runs.
        # The space of packings is large.
        
        return triples

    # Run multiple random greedy trials
    trials = 0
    while time.time() < deadline:
        trials += 1
        
        # Random greedy
        trs = try_random_greedy()
        
        # Calculate alpha
        alpha = get_alpha(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs
            
            # If we find a very good alpha, we can stop early?
            # Theoretical lower bound for alpha?
            # For a 2-(16,3,1) packing, the maximum number of triples is floor(16/3 * 15/2) = 40.
            # Actually, the maximum size of a 2-(16,3,1) packing is 40 (since 16*15/6 = 40).
            # So a Steiner Triple System STS(16) does not exist (16 is not 1 or 3 mod 6).
            # Max triples = 40.
            # If we have 40 triples, what is the alpha?
            
            # If best_alpha is low enough, we can break.
            # For v=16, a good packing should have alpha around 6-8?
            # Let's aim for <= 7.
            if best_alpha <= 6:
                break

    # If we have time left, try to improve the best solution found with local search?
    # Or just return the best found.
    
    # Ensure we return a valid list of triples
    # If no triples found (unlikely), return empty list?
    # The task requires a list of triples.
    
    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
