import time
import random
import itertools
from collections import defaultdict

# Deadline: evaluator kills at 150s wall-clock. Use 140s as safety margin.
DEADLINE = 140.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Helper functions defined inside to keep interface clean
    def is_valid_system(triples):
        """Check that any two triples share at most one point."""
        n = len(triples)
        for i in range(n):
            for j in range(i + 1, n):
                shared = set(triples[i]) & set(triples[j])
                if len(shared) > 1:
                    return False
        return True
    
    def calc_alpha(triples):
        """Calculate independence number (largest set containing no triple)."""
        n = 16
        # Build a set of triples as frozensets for fast lookup
        triple_set = set(frozenset(t) for t in triples)
        
        # Use bitmask approach: a subset S is independent if no triple is subset of S
        # We want max |S| such that no triple is subset of S
        # Equivalently: for each triple, at least one point is NOT in S
        # This is a hitting set / covering problem on the complement
        # Actually: alpha = max |S| such that for all triples T, T not subset of S
        # Which means: for all triples T, (T \ S) is non-empty, i.e., S does not contain T
        # Complement: let C = [0..15] \ S, then for all triples T, T ∩ C is non-empty
        # So C is a hitting set for the hypergraph of triples
        # alpha = 16 - min hitting set size
        # min hitting set size = minimum number of points to hit all triples
        
        # For 16 points and ~20-30 triples, we can try all subsets of small size for hitting set
        # Or use the direct approach: iterate over all subsets of size k from high to low
        
        # Since we want to minimize alpha, and alpha is typically 6-8 for good systems,
        # we can check: does there exist an independent set of size 7? 8? etc.
        
        # Check if there's an independent set of size k
        # An independent set S of size k means no triple is entirely in S
        # Number of subsets of size k from 16: C(16,8)=12870, C(16,9)=11440, C(16,10)=8008
        # For each subset, check if any triple is contained - this is fast with bitmasks
        
        # Represent each triple as a bitmask
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        # Check for independent set of size k
        def has_independent_set_of_size(k):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is subset of this mask
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:  # triple is contained in subset
                        independent = False
                        break
                if independent:
                    return True
            return False
        
        # Binary search or linear search from high k down
        for k in range(16, 0, -1):
            if has_independent_set_of_size(k):
                return k
        return 0
    
    def calc_alpha_fast(triples):
        """Faster alpha calculation using precomputed masks."""
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        # For a given subset mask, check if it's independent (no triple contained)
        # We'll iterate k from high to low
        
        # Precompute all subset masks for each k? No, too many.
        # Instead, for each k, iterate combinations and check
        
        for k in range(16, 5, -1):  # We know alpha is at least 5 for good systems
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        # Fallback: check smaller k
        for k in range(5, 0, -1):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    def try_improve(triples, time_limit=10.0):
        """Local search: swap points between triples to reduce alpha."""
        current_alpha = calc_alpha_fast(triples)
        best_alpha = current_alpha
        best_triples = [list(t) for t in triples]
        
        deadline = time.time() + time_limit
        no_improve_streak = 0
        
        while time.time() < deadline:
            improved = False
            # Try random swaps: pick two triples and swap one point each
            for _ in range(200):
                if time.time() >= deadline:
                    break
                i, j = random.sample(range(len(triples)), 2)
                # Pick positions
                pos_i = random.randint(0, 2)
                pos_j = random.randint(0, 2)
                
                p1 = triples[i][pos_i]
                p2 = triples[j][pos_j]
                if p1 == p2:
                    continue
                
                # Swap
                triples[i][pos_i] = p2
                triples[j][pos_j] = p1
                
                # Check validity
                if is_valid_system(triples):
                    new_alpha = calc_alpha_fast(triples)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = [list(t) for t in triples]
                        improved = True
                        no_improve_streak = 0
                    else:
                        # Revert
                        triples[i][pos_i] = p1
                        triples[j][pos_j] = p2
                        no_improve_streak += 1
                else:
                    # Revert
                    triples[i][pos_i] = p1
                    triples[j][pos_j] = p2
                    no_improve_streak += 1
            
            if no_improve_streak > 500:
                # Random restart with perturbation
                for _ in range(10):
                    i, j = random.sample(range(len(triples)), 2)
                    pos_i = random.randint(0, 2)
                    pos_j = random.randint(0, 2)
                    p1 = triples[i][pos_i]
                    p2 = triples[j][pos_j]
                    triples[i][pos_i] = p2
                    triples[j][pos_j] = p1
                    if is_valid_system(triples):
                        break
                    else:
                        triples[i][pos_i] = p1
                        triples[j][pos_j] = p2
                no_improve_streak = 0
            
            if best_alpha <= 5:
                break
        
        return best_triples, best_alpha
    
    # Start with a good initial structure
    # STS(15) on points 0-14, then add point 15
    # A known STS(15) structure (Kirkman's schoolgirl problem related)
    # Let's build a reasonable starting point
    
    # Points 0-14: use a Z_5 x Z_3 structure or similar
    # Actually, let's just use a heuristic initial construction and improve
    
    # Initial: create triples covering all 16 points
    # Start with a near-STC structure
    
    # Use a known STS(15) on 0-14, then add triples with 15
    # STS(15) has 35 triples. We'll use a subset and add 15.
    
    # Let's try a different approach: start with random valid system and improve
    
    best_overall = None
    best_alpha_overall = 999
    
    for attempt in range(5):
        if time.time() > DEADLINE - 30:
            break
        
        # Generate initial system
        # Try to create a system with ~20-25 triples
        triples = []
        used_pairs = set()
        
        points = list(range(16))
        random.shuffle(points)
        
        # Greedily add triples
        remaining = set(range(16))
        while len(remaining) >= 3 and time.time() < DEADLINE - 20:
            # Pick 3 random remaining points
            candidates = list(remaining)
            random.shuffle(candidates)
            added = False
            for i in range(min(3, len(candidates))):
                for j in range(i+1, min(3, len(candidates))):
                    # Try triple with first point and two others
                    pass
            
            # Just pick 3 random
            if len(remaining) >= 3:
                t = random.sample(list(remaining), 3)
                t.sort()
                # Check pair uniqueness
                pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                if all(p not in used_pairs for p in pairs):
                    triples.append(t)
                    used_pairs.update(pairs)
                    remaining -= set(t)
                    added = True
            
            if not added:
                # Try to find any valid triple from remaining
                found = False
                rem_list = list(remaining)
                for a in range(len(rem_list)):
                    if found: break
                    for b in range(a+1, len(rem_list)):
                        if found: break
                        for c in range(b+1, len(rem_list)):
                            t = sorted([rem_list[a], rem_list[b], rem_list[c]])
                            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                            if all(p not in used_pairs for p in pairs):
                                triples.append(t)
                                used_pairs.update(pairs)
                                remaining -= set(t)
                                found = True
                                break
        
        if len(triples) < 5:
            continue
        
        # Check validity
        if not is_valid_system(triples):
            continue
        
        alpha = calc_alpha_fast(triples)
        if alpha < best_alpha_overall:
            best_alpha_overall = alpha
            best_overall = [list(t) for t in triples]
        
        # Improve
        if time.time() < DEADLINE - 25:
            time_left = DEADLINE - 25 - time.time()
            improved_triples, improved_alpha = try_improve([list(t) for t in triples], time_left)
            if improved_alpha < best_alpha_overall:
                best_alpha_overall = improved_alpha
                best_overall = improved_triples
        
        if best_alpha_overall <= 5:
            break
    
    # If no good result found, use fallback
    if best_overall is None:
        # Fallback: the original seed structure
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_overall
