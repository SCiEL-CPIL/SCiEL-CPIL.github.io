import random
import itertools
from functools import lru_cache

# Precompute masks for all subsets of {0..15}, sorted by size descending
# This allows us to find the largest independent set quickly by checking
# large subsets first.
_SUBSET_MASKS = []
for i in range(1 << 16):
    _SUBSET_MASKS.append(i)
_SUBSET_MASKS.sort(key=lambda x: x.bit_count(), reverse=True)

def calculate_independence_number(triples):
    """
    Calculate the size of the largest subset of {0..15} that contains no triple from 'triples'.
    Uses precomputed sorted masks to check from largest size down.
    """
    if not triples:
        return 16
    
    # Precompute masks for the given triples
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check subsets from largest size down
    # We iterate through precomputed masks sorted by bit_count descending
    for m in _SUBSET_MASKS:
        # Optimization: if we've already checked all masks of this size, 
        # and none were independent, we can stop if we want to return the 
        # size of the first one found. 
        # But we need to check ALL masks of a given size to be sure none are independent?
        # No, if we find ONE independent set of size k, the answer is k.
        # So we just need to find the first mask in the sorted list that is independent.
        
        is_independent = True
        for tm in triple_masks:
            if (m & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            return m.bit_count()
            
    return 0

def greedy_construct(seed):
    random.seed(seed)
    points = list(range(16))
    # Generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple to speed up conflict checking
    triple_data = []
    for t in all_triples:
        pairs = ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))
        triple_data.append((t, pairs))
    
    selected_triples = []
    covered_pairs = set()
    
    # Max possible triples is 120 pairs / 3 pairs per triple = 40
    # We iterate until no more valid triples can be added
    
    while True:
        # Find all valid triples
        valid_candidates = []
        for t, pairs in triple_data:
            # Check if any pair is already covered
            conflict = False
            for p in pairs:
                if p in covered_pairs:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append((t, pairs))
        
        if not valid_candidates:
            break
        
        # Strategy: Evaluate a sample of valid candidates to find the one
        # that minimizes the independence number.
        # If the number of candidates is small, check all.
        # Otherwise, sample a random subset to keep computation time low.
        
        sample_size = min(len(valid_candidates), 30)
        if len(valid_candidates) > sample_size:
            candidates_to_check = random.sample(valid_candidates, sample_size)
        else:
            candidates_to_check = valid_candidates
        
        best_t = None
        best_pairs = None
        best_score = 17 # Max is 16
        
        for t, pairs in candidates_to_check:
            # Temporarily add t
            new_triples = selected_triples + [t]
            score = calculate_independence_number(new_triples)
            
            if score < best_score:
                best_score = score
                best_t = t
                best_pairs = pairs
                
            # Early termination if we found a very good score
            # The minimum possible independence number for 16 points is likely small.
            # If we get 2 or 3, it's probably optimal or near optimal.
            if best_score <= 3:
                break
        
        if best_t is None:
            break
            
        selected_triples.append(best_t)
        for p in best_pairs:
            covered_pairs.add(p)
            
    return selected_triples

def construct():
    best_triples = []
    best_score = 17 # Max possible is 16, so 17 is infinity
    
    # Run multiple greedy attempts
    # 20 seeds should be fast enough with the optimized independence check
    for seed in range(20):
        triples = greedy_construct(seed)
        score = calculate_independence_number(triples)
        if score < best_score:
            best_score = score
            best_triples = triples
            
    return best_triples
