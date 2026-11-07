import time
import random

def construct():
    n = 128
    # Use bitset for speed
    mask = (1 << (n + 1)) - 2  # bits 1..128 set
    
    def get_elems(m):
        return [i for i in range(1, n + 1) if (m >> i) & 1]
    
    def find_bad_triple(m):
        elems = get_elems(m)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                if a + b > n: continue
                if not ((m >> (a + b)) & 1): continue
                for k in range(j + 1, len(elems)):
                    c = elems[k]
                    if b + c > n: break
                    if not ((m >> (b + c)) & 1): continue
                    if not ((m >> (a + c)) & 1): continue
                    return (a, b, c)
        return None

    def count_conflicts(m):
        elems = get_elems(m)
        counts = {x: 0 for x in elems}
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                if a + b > n: continue
                if not ((m >> (a + b)) & 1): continue
                for k in range(j + 1, len(elems)):
                    c = elems[k]
                    if b + c > n: break
                    if not ((m >> (b + c)) & 1): continue
                    if not ((m >> (a + c)) & 1): continue
                    for val in [a, b, c, a+b, a+c, b+c]:
                        if (m >> val) & 1:
                            counts[val] += 1
        return counts

    start_time = time.time()
    deadline = start_time + 55

    # Phase 1: Greedy Removal based on conflict count
    while time.time() < deadline:
        # Check if valid
        if find_bad_triple(mask) is None:
            break
        
        # Count conflicts to find best removal candidate
        counts = count_conflicts(mask)
        if not counts:
            break
            
        max_elem = max(counts, key=counts.get)
        mask &= ~(1 << max_elem)
        
    # Phase 2: Local Search to add elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [i for i in range(1, n + 1) if not ((mask >> i) & 1)]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_mask = mask | (1 << x)
            if find_bad_triple(new_mask) is None:
                mask = new_mask
                improved = True
                
    # Phase 3: Try swaps to potentially unlock additions?
    # Or just return.
    
    return [i for i in range(1, n + 1) if (mask >> i) & 1]
