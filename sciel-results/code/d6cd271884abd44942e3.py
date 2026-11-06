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
    
    for trial in range(50):
        random.seed(trial)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        for idx in indices:
            c = all_sets[idx]
            conflict = False
            for i in range(len(chosen)):
                a = chosen[i]
                ab = a & c
                # Check if there exists j < i such that chosen[j] & c == ab and chosen[i] & chosen[j] == ab
                # Actually, we need: for some pair (a, b) in chosen, a&b == a&c == b&c
                # So for this c and a, compute ac = a&c. Then we need some b in chosen (b != a) such that b&c == ac and a&b == ac.
                # This is equivalent to: b contains ac, b is disjoint from a\ac, and b is disjoint from c\ac.
                # But we can just check: for each b in chosen, if a&b == ac and b&c == ac, then conflict.
                # That's O(k) per a, so O(k^2) total.
                pass
            
            # Let me just do the O(k^2) check directly
            k = len(chosen)
            for i in range(k):
                a = chosen[i]
                ac = a & c
                for j in range(i + 1, k):
                    b = chosen[j]
                    if (a & b) == ac and (b & c) == ac:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen.append(c)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elems)
    
    return result
