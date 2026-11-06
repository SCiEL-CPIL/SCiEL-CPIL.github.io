import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    best_size = 0
    best_family = []
    
    # Helper function to check if candidate C conflicts with family F
    def is_conflict(c, F):
        # Group sets in F by their intersection with c
        groups = {}
        for a in F:
            inter = a & c
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(a)
        
        for inter, group in groups.items():
            # Check if any pair in group has intersection exactly 'inter'
            # i.e., (a ^ inter) & (b ^ inter) == 0
            k = len(group)
            if k < 2:
                continue
            
            # Optimization for small k or specific inter sizes
            # Since |a|=4, |inter| <= 3.
            # If |inter| == 3, a^inter has 1 bit.
            # If |inter| == 2, a^inter has 2 bits.
            
            # General check
            for i in range(k):
                a = group[i]
                a_rem = a ^ inter
                for j in range(i + 1, k):
                    b = group[j]
                    b_rem = b ^ inter
                    if (a_rem & b_rem) == 0:
                        return True
        return False

    for trial in range(100):
        random.seed(trial)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        for idx in indices:
            c = all_sets[idx]
            if not is_conflict(c, chosen):
                chosen.append(c)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # Early stop if we hit a high target?
            # Max possible is unknown, but 250 is the cap.
            # Let's just run all trials.
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elems)
    
    return result
