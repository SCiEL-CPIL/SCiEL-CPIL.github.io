import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks = {}
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = m
        
    # Helper to check linearity: two triples share at most 1 point
    def is_linear(t1, t2):
        # t1, t2 are tuples
        # Intersection size <= 1
        # Using sets is slow, use masks or simple check
        # Since n=16, sets are okay, but let's use bit counts
        m1 = triple_masks[t1]
        m2 = triple_masks[t2]
        inter = m1 & m2
        # Count bits
        count = 0
        while inter:
            inter &= inter - 1
            count += 1
        return count <= 1

    def compute_alpha(triples):
        # Find largest subset with no triple
        # Check sizes 16 down to 1
        # We can optimize by checking only sizes 8, 7, 6...
        # If we find an independent set of size k, return k (or keep searching for larger? No, we want max)
        # We want the MAX size.
        # So check 16, 15, ...
        # But 16 is unlikely. 
        # Let's check 8, 7, 6, 5, 4, 3, 2, 1
        # If no independent set of size 8, then alpha <= 7.
        # We need the exact max to minimize it.
        
        # Precompute masks of current triples
        t_masks = [triple_masks[t] for t in triples]
        
        # Check size k
        def has_independent_set(k):
            for s in itertools.combinations(points, k):
                mask_s = 0
                for p in s:
                    mask_s |= (1 << p)
                # Check if any triple is subset of s
                # i.e., mask_t & mask_s == mask_t
                is_indep = True
                for mt in t_masks:
                    if (mt & mask_s) == mt:
                        is_indep = False
                        break
                if is_indep:
                    return True
            return False

        for k in range(16, 0, -1):
            if has_independent_set(k):
                return k
        return 1

    # Greedy initial packing
    def greedy_packing():
        current = []
        # Randomize order for diversity
        random.shuffle(all_triples)
        for t in all_triples:
            can_add = True
            for existing in current:
                if not is_linear(t, existing):
                    can_add = False
                    break
            if can_add:
                current.append(t)
        return current

    best_packing = greedy_packing()
    best_alpha = compute_alpha(best_packing)
    
    # Local Search
    # Try to improve by replacing triples
    # A move: Remove a triple, add a different one.
    # Or: Remove a set of triples, add a set of triples.
    
    # Simpler: Random restarts and local search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Random restart every 100 iterations or so?
        # Or just perturb current best.
        
        # Perturb: Remove 1-3 random triples, then greedily add random valid triples
        current = list(best_packing)
        if random.random() < 0.1:
            # Full restart
            current = greedy_packing()
        else:
            # Local perturbation
            n_remove = random.randint(1, 3)
            for _ in range(n_remove):
                if current:
                    idx = random.randint(0, len(current) - 1)
                    current.pop(idx)
            
            # Greedily add back
            candidates = list(all_triples)
            random.shuffle(candidates)
            for t in candidates:
                can_add = True
                for existing in current:
                    if not is_linear(t, existing):
                        can_add = False
                        break
                if can_add:
                    current.append(t)
                    # Stop if we hit max possible? Max is 37.
                    # Just fill up.
        
        alpha = compute_alpha(current)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_packing = current
            # If we reach alpha 6, that's likely optimal or close.
            # Lower bound for alpha?
            # With 37 triples, each triple covers 3 points.
            # Total incidences 111.
            # If alpha=6, no 6-set is independent.
            # This is a strong condition.
            
            # If alpha <= 6, we can stop early? 
            # Is alpha 5 possible?
            # Let's keep searching a bit more to be sure, or stop if time allows.
            if best_alpha <= 6:
                # Try to improve to 5?
                # If we find 6, it's good. 1.0 seed was 11. 1.375 was likely alpha 7 or 8?
                # Score is seed/value. Seed=1.0. Value=alpha? No, "score is the largest point set... Minimize".
                # "search score is seed/value".
                # If value=alpha, then score = 1.0 / alpha.
                # If alpha=8, score=0.125? No, "1.0 equals the seed program".
                # Seed program: 5 disjoint triples. Alpha = 11 (points 0..14? No, 5*3=15 points used. Point 15 is alone. 
                # Wait, seed: [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]].
                # Points 0..14 are in triples. Point 15 is not.
                # Subset {15} is independent.
                # Subset {15, 14, 13, ...} 
                # Any subset of the 11 points not in any triple is independent.
                # Points not in any triple: 15.
                # Wait, the triples are disjoint.
                # Points 0..14 are covered. 15 is not.
                # A set is independent if it contains NO triple.
                # Since triples are disjoint, a set containing one full triple is NOT independent.
                # A set containing parts of triples is independent if it doesn't contain any full triple.
                # Max independent set: Take 2 points from each triple (5*2=10) + point 15 = 11 points.
                # So alpha = 11.
                # Score = 1.0 / 11? No, "1.0 equals the seed program".
                # Maybe the score is defined differently?
                # "The score is the largest point set... Minimize."
                # "search score is seed/value, so 1.0 equals the seed program".
                # This implies Value(seed) = Seed_Score / 1.0 ?
                # Or maybe the "value" is the alpha, and the "score" is inverted?
                # If Seed Alpha = 11, and Score = 1.0, then Score = 11 / Alpha?
                # If Alpha = 8, Score = 11/8 = 1.375.
                # This matches the observation "1 -> 1.375" for a program that improved island 0.
                # So Score = 11 / Alpha.
                # We want to maximize Score, which means minimize Alpha.
                # If we find Alpha = 6, Score = 11/6 = 1.833.
                # If we find Alpha = 5, Score = 11/5 = 2.2.
                
                # So minimizing Alpha is the goal.
                pass

    return best_packing

# Quick check on logic:
# Seed Alpha = 11.
# Improved Alpha = 8 (Score 1.375 = 11/8).
# Target Alpha < 8.
