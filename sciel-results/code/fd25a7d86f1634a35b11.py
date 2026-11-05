"""Linear triple systems on 16 points minimizing independence number."""
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

def _independence_number(triples):
    """Exact independence number: largest subset with no triple fully contained."""
    n = 16
    # Precompute: for each point, bitmask of points it's in a triple with
    # A subset S is independent if no triple is fully contained in S
    # Equivalently: for every triple [a,b,c], S doesn't contain all of a,b,c
    
    # Use bitmask: triple_mask[t] = (1<<a)|(1<<b)|(1<<c)
    triple_masks = [0] * len(triples)
    for i, (a, b, c) in enumerate(triples):
        triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
    
    # Check all subsets by size, from largest down
    # For n=16, there are 2^16 = 65536 subsets
    # Check from size 16 down to 1
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _independence_number_fast(triples):
    """Faster independence number using precomputed triple masks and subset enumeration."""
    n = 16
    triple_masks = []
    for (a, b, c) in triples:
        triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    # Precompute for each subset whether it's independent
    # Use DP: independent[mask] = no triple fully contained in mask
    # A triple is contained in mask if (mask & tm) == tm
    # For efficiency, check subsets from largest size down
    
    # For each subset mask (0 to 2^16-1), check independence
    # But we want the largest independent set
    # Enumerate by size
    
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    indep = False
                    break
            if indep:
                return size
    return 0

def _has_independent_set_of_size(triples, size):
    """Check if there exists an independent set of given size. Returns True if found."""
    n = 16
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in triples]
    
    for subset in itertools.combinations(range(n), size):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        indep = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                indep = False
                break
        if indep:
            return True
    return False

def _is_valid(triples):
    """Check linearity: any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _count_pairs(triples):
    """Count total pair occurrences (each triple contributes 3 pairs)."""
    return 3 * len(triples)

def _greedy_construct(n_points=16, max_triples=40):
    """Greedy construction of a linear triple system."""
    random.seed(42)
    triples = []
    used_pairs = set()
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n_points), 3))
    random.shuffle(all_triples)
    
    for t in all_triples:
        if len(triples) >= max_triples:
            break
        # Check if adding this triple maintains linearity
        valid = True
        new_pairs = set()
        for i in range(3):
            for j in range(i + 1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                new_pairs.add(pair)
        
        # Check against existing pairs
        for pair in new_pairs:
            if pair in used_pairs:
                valid = False
                break
        
        if valid:
            triples.append(list(t))
            used_pairs.update(new_pairs)
    
    return triples

def _local_search(triples, time_limit=120.0):
    """Local search to minimize independence number."""
    start_time = time.time()
    best_triples = list(triples)
    best_alpha = _independence_number_fast(best_triples)
    
    if best_alpha <= 7:
        return best_triples, best_alpha
    
    # Local search: try swapping triples or adding/removing
    n_points = 16
    all_triples = list(itertools.combinations(range(n_points), 3))
    
    def _is_linear(trip_list):
        for i in range(len(trip_list)):
            for j in range(i + 1, len(trip_list)):
                common = set(trip_list[i]) & set(trip_list[j])
                if len(common) > 1:
                    return False
        return True
    
    # Strategy: iteratively try to replace triples to reduce alpha
    # Focus on the independent set of size best_alpha and try to "hit" it
    
    for iteration in range(1000000):
        if time.time() - start_time > time_limit:
            break
        
        # Find current independent set
        alpha = best_alpha
        # Try to find an independent set of size alpha
        indep_set = None
        for subset in itertools.combinations(range(n_points), alpha):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            indep = True
            for (a, b, c) in best_triples:
                tm = (1 << a) | (1 << b) | (1 << c)
                if (mask & tm) == tm:
                    indep = False
                    break
            if indep:
                indep_set = set(subset)
                break
        
        if indep_set is None:
            # No independent set of size alpha found, recompute
            best_alpha = _independence_number_fast(best_triples)
            alpha = best_alpha
            if best_alpha <= 7:
                break
            continue
        
        # Try to add a triple that hits the independent set
        # A triple hits the set if it contains at least one point from indep_set
        # But we need to maintain linearity
        # Better: try to add a triple that contains 2+ points from indep_set
        # to reduce the independent set size
        
        # Try adding a new triple
        added = False
        for t in all_triples:
            t_set = set(t)
            # Check if it hits the independent set with at least 1 point
            if not (t_set & indep_set):
                continue
            
            # Check linearity with existing triples
            valid = True
            for existing in best_triples:
                common = t_set & set(existing)
                if len(common) > 1:
                    valid = False
                    break
            
            if valid:
                new_triples = best_triples + [list(t)]
                new_alpha = _independence_number_fast(new_triples)
                if new_alpha < best_alpha:
                    best_triples = new_triples
                    best_alpha = new_alpha
                    added = True
                    break
                elif new_alpha == best_alpha:
                    # Try it anyway for diversity
                    if random.random() < 0.1:
                        best_triples = new_triples
                        added = True
                        break
        
        if not added:
            # Try removing a triple that's not helping and adding a better one
            # Or try swapping
            if random.random() < 0.5 and len(best_triples) > 0:
                # Remove a random triple and try to add two better ones
                idx = random.randint(0, len(best_triples) - 1)
                removed = best_triples.pop(idx)
                
                # Try to add two triples
                for t1 in all_triples:
                    t1_set = set(t1)
                    valid1 = True
                    for existing in best_triples:
                        if len(t1_set & set(existing)) > 1:
                            valid1 = False
                            break
                    if valid1:
                        best_triples.append(list(t1))
                        for t2 in all_triples:
                            t2_set = set(t2)
                            valid2 = True
                            for existing in best_triples:
                                if len(t2_set & set(existing)) > 1:
                                    valid2 = False
                                    break
                            if valid2:
                                best_triples.append(list(t2))
                                new_alpha = _independence_number_fast(best_triples)
                                if new_alpha < best_alpha:
                                    best_alpha = new_alpha
                                break
                        break
                else:
                    # Revert
                    best_triples.append(removed)
        
        if best_alpha <= 7:
            break
    
    return best_triples, best_alpha

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    
    # Phase 1: Greedy construction
    triples = _greedy_construct()
    
    # Phase 2: Local search with deadline guard
    remaining_time = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining_time > 5:
        triples, alpha = _local_search(triples, time_limit=min(remaining_time, 130.0))
    else:
        alpha = _independence_number_fast(triples)
    
    # Verify validity
    if not _is_valid(triples):
        # Fallback to simple valid system
        triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return triples

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
