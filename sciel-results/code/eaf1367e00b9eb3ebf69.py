"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s
    
    random.seed(42)
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        n = len(triples)
        for i in range(n):
            si = set(triples[i])
            for j in range(i + 1, n):
                sj = set(triples[j])
                if len(si & sj) > 1:
                    return False
        return True
    
    def used_pairs(triples):
        """Return set of all pairs that appear together in some triple."""
        pairs = set()
        for t in triples:
            a, b, c = t
            pairs.add((min(a,b), max(a,b)))
            pairs.add((min(a,c), max(a,c)))
            pairs.add((min(b,c), max(b,c)))
        return pairs
    
    def can_add(triples, new_triple, pair_set):
        """Check if new_triple can be added without violating linearity."""
        a, b, c = new_triple
        for p in [(a,b), (a,c), (b,c)]:
            p = (min(p), max(p))
            if p in pair_set:
                return False
        return True
    
    def independence_number(triples):
        """Find the largest independent set (no triple fully contained)."""
        # Use bitmask approach
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        # Try to find largest independent set by checking subsets
        # For 16 points, we can use a smarter approach
        # An independent set is one that contains no triple
        # We want the maximum size such set
        
        best_size = 0
        # Use recursive search with pruning
        # Sort points by degree (how many triples they're in)
        point_triple_count = [0] * 16
        for m in triple_masks:
            for p in range(16):
                if m & (1 << p):
                    point_triple_count[p] += 1
        
        # Greedy + local search for independence number
        # Start with all points, remove to break triples
        # Actually, we want to FIND the largest set with no triple
        # This is equivalent to: max |S| such that no triple is subset of S
        
        # Use a different approach: iterate over all subsets is 2^16 = 65536
        # That's fast enough!
        for mask in range(1 << 16):
            # Check if this subset contains any triple
            has_triple = False
            for tm in triple_masks:
                if (mask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                size = bin(mask).count('1')
                if size > best_size:
                    best_size = size
                    if best_size >= 13:  # early exit if we already have large
                        break
        
        return best_size
    
    def find_largest_independent_set(triples):
        """Return the actual largest independent set as a bitmask."""
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        best_mask = 0
        best_size = 0
        for mask in range(1 << 16):
            has_triple = False
            for tm in triple_masks:
                if (mask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                size = bin(mask).count('1')
                if size > best_size:
                    best_size = size
                    best_mask = mask
        
        return best_mask, best_size
    
    def generate_initial_system():
        """Generate a random linear triple system."""
        triples = []
        pair_set = set()
        points = list(range(16))
        
        # Greedy: add random triples
        attempts = 0
        max_attempts = 10000
        while attempts < max_attempts:
            attempts += 1
            a, b, c = random.sample(points, 3)
            new_triple = [a, b, c]
            if can_add(triples, new_triple, pair_set):
                triples.append(new_triple)
                pair_set.add((min(a,b), max(a,b)))
                pair_set.add((min(a,c), max(a,c)))
                pair_set.add((min(b,c), max(b,c)))
        
        return triples
    
    def local_search_improve(triples, time_limit):
        """Improve the system by targeting the largest independent set."""
        end_time = start_time + time_limit
        best_triples = triples[:]
        best_alpha = independence_number(triples)
        
        while time.time() < end_time:
            # Find largest independent set
            indep_mask, indep_size = find_largest_independent_set(best_triples)
            
            if indep_size <= best_alpha:
                best_alpha = indep_size
            
            # Try to break this independent set by modifying triples
            # Strategy: find a triple that we can modify to include a point from the independent set
            # while maintaining linearity
            
            improved = False
            indep_points = [p for p in range(16) if indep_mask & (1 << p)]
            non_indep_points = [p for p in range(16) if not (indep_mask & (1 << p))]
            
            # Try replacing a triple to "cover" more of the independent set
            # Or add a new triple that includes points from the independent set
            pair_set = used_pairs(best_triples)
            
            # Try to add a triple that includes at least one point from indep set
            # and helps break the independence
            random.shuffle(indep_points)
            random.shuffle(non_indep_points)
            
            for _ in range(200):
                if time.time() >= end_time:
                    break
                
                # Try adding a new triple
                if len(non_indep_points) >= 2 and len(indep_points) >= 1:
                    # Triple with 1 indep + 2 non-indep
                    a = random.choice(indep_points)
                    b, c = random.sample(non_indep_points, 2)
                    new_triple = [a, b, c]
                    if can_add(best_triples, new_triple, pair_set):
                        # Check if this improves
                        test_triples = best_triples + [new_triple]
                        test_alpha = independence_number(test_triples)
                        if test_alpha < best_alpha:
                            best_triples = test_triples
                            best_alpha = test_alpha
                            pair_set = used_pairs(best_triples)
                            improved = True
                            break
                elif len(indep_points) >= 3:
                    # Triple with 3 indep points - this directly breaks the set
                    a, b, c = random.sample(indep_points, 3)
                    new_triple = [a, b, c]
                    if can_add(best_triples, new_triple, pair_set):
                        test_triples = best_triples + [new_triple]
                        test_alpha = independence_number(test_triples)
                        if test_alpha < best_alpha:
                            best_triples = test_triples
                            best_alpha = test_alpha
                            pair_set = used_pairs(best_triples)
                            improved = True
                            break
                
                # Try modifying an existing triple
                if not improved and best_triples:
                    idx = random.randrange(len(best_triples))
                    old_triple = best_triples[idx]
                    # Try to replace one point in old_triple with an indep point
                    for i in range(3):
                        if len(indep_points) < 1:
                            break
                        new_point = random.choice(indep_points)
                        if new_point in old_triple:
                            continue
                        new_triple = old_triple[:]
                        new_triple[i] = new_point
                        if len(set(new_triple)) != 3:
                            continue
                        
                        # Check if we can swap
                        test_triples = best_triples[:]
                        test_triples[idx] = new_triple
                        if is_linear(test_triples):
                            test_alpha = independence_number(test_triples)
                            if test_alpha < best_alpha:
                                best_triples = test_triples
                                best_alpha = test_alpha
                                pair_set = used_pairs(best_triples)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Try a different strategy: rebuild around the independent set
                # Remove some triples and add better ones
                if len(best_triples) > 5:
                    # Remove a random triple
                    idx = random.randrange(len(best_triples))
                    removed = best_triples.pop(idx)
                    pair_set = used_pairs(best_triples)
                    
                    # Try to add 1-2 new triples that target the independent set
                    added_count = 0
                    for _ in range(10):
                        if time.time() >= end_time:
                            break
                        if len(indep_points) >= 2 and len(non_indep_points) >= 1:
                            a, b = random.sample(indep_points, 2)
                            c = random.choice(non_indep_points)
                            new_triple = [a, b, c]
                            if can_add(best_triples, new_triple, pair_set):
                                best_triples.append(new_triple)
                                pair_set.add((min(a,b), max(a,b)))
                                pair_set.add((min(a,c), max(a,c)))
                                pair_set.add((min(b,c), max(b,c)))
                                added_count += 1
                        elif len(indep_points) >= 3:
                            a, b, c = random.sample(indep_points, 3)
                            new_triple = [a, b, c]
                            if can_add(best_triples, new_triple, pair_set):
                                best_triples.append(new_triple)
                                pair_set.add((min(a,b), max(a,b)))
                                pair_set.add((min(a,c), max(a,c)))
                                pair_set.add((min(b,c), max(b,c)))
                                added_count += 1
                        
                        if added_count >= 2:
                            break
                    
                    # Check if improved
                    test_alpha = independence_number(best_triples)
                    if test_alpha < best_alpha:
                        best_alpha = test_alpha
                    else:
                        # Revert
                        best_triples.insert(idx, removed)
        
        return best_triples, best_alpha
    
    # Generate multiple initial systems and pick the best
    best_system = None
    best_alpha = 16
    
    # First, try a structured construction
    # Use affine geometry idea: points as F_2^4
    # Create triples from 2D affine subspaces (4-element flats)
    # But we need triples (3 elements), so we can take subsets
    
    # Actually, let's try: for each 2D subspace (4 points), add all 4 triples (combinations of 3)
    # But that might violate linearity... let's check
    # Two different 2D subspaces in F_2^4 can share at most 1 point (if they're not the same)
    # Wait, two distinct 2D subspaces through origin can share a 1D subspace (2 points)
    # So we need to be careful
    
    # Let's just use randomized greedy with local search
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        triples = generate_initial_system()
        alpha = independence_number(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = triples[:]
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            improved_triples, improved_alpha = local_search_improve(triples, remaining_time - 2)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_system = improved_triples
    
    if best_system is None:
        best_system = generate_initial_system()
    
    return best_system
# EVOLVE-BLOCK-END
