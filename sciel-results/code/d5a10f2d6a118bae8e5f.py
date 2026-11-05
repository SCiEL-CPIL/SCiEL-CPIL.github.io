import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denoms = list(range(2, 49))
    n = len(denoms)
    vals_float = [1.0 / d for d in denoms]
    vals_frac = [Fraction(1, d) for d in denoms]
    
    best_deficit = None
    best_indices = None
    best_sum_frac = None
    
    # Helper to update best
    def try_update(indices):
        nonlocal best_deficit, best_indices, best_sum_frac
        # Calculate exact sum
        s = Fraction(0)
        for idx in indices:
            s += vals_frac[idx]
        if s < 1:
            deficit = 1 - s
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = indices
                best_sum_frac = s

    # 1. Check known good candidate
    # Denoms 2, 3, 7, 43 -> Indices 0, 1, 5, 41
    # Sum = 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806
    # Deficit = 1/1806
    known_candidate = [0, 1, 5, 41]
    try_update(known_candidate)
    
    # Dynamic threshold
    # If best_deficit is found, we only care about sums > 1 - best_deficit
    # To avoid checking too many, we use a slightly looser threshold initially,
    # but since we have a very good candidate (1/1806 ~ 0.00055),
    # we can set the threshold close to 1.
    
    # Let's use a fixed threshold of 1 - 0.0001 for the search, 
    # effectively looking for deficits < 0.0001.
    # If none found, the known candidate stands.
    # If found, it's better.
    
    search_threshold = 1.0 - 0.0001
    
    # 2. Exhaustive search for sizes 3 to 6
    # We iterate sizes 3, 4, 5, 6
    
    for size in [3, 4, 5, 6]:
        for comb in combinations(range(n), size):
            if time.time() > deadline:
                return best_indices if best_indices else [0]
            
            # Quick float check
            # Unroll sum for speed?
            if size == 3:
                s_float = vals_float[comb[0]] + vals_float[comb[1]] + vals_float[comb[2]]
            elif size == 4:
                s_float = vals_float[comb[0]] + vals_float[comb[1]] + vals_float[comb[2]] + vals_float[comb[3]]
            elif size == 5:
                s_float = vals_float[comb[0]] + vals_float[comb[1]] + vals_float[comb[2]] + vals_float[comb[3]] + vals_float[comb[4]]
            else: # size == 6
                s_float = vals_float[comb[0]] + vals_float[comb[1]] + vals_float[comb[2]] + vals_float[comb[3]] + vals_float[comb[4]] + vals_float[comb[5]]
                
            if s_float >= 1.0:
                continue
                
            if s_float > search_threshold:
                try_update(list(comb))
                
                # If we found a deficit smaller than 1/1806, we can tighten threshold
                if best_deficit is not None and best_deficit < Fraction(1, 1806):
                    search_threshold = 1.0 - float(best_deficit) - 1e-9 # Slightly looser to catch equal? No, we want strictly better.
                    # Actually, if we find one, we want to find even better ones.
                    # So threshold should be 1 - new_best_deficit.
                    # But float precision... 
                    # Let's just keep it simple. The known candidate is very good.
                    
    if best_indices is None:
        return [0]
        
    return best_indices

if __name__ == "__main__":
    construct()
