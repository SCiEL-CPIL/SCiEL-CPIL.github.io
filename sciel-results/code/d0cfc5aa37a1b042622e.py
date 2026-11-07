import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    
    # Precompute comparability table for all 2^12 masks?
    # 4096 * 4096 is 16M entries. Too big for memory if stored as bool matrix? 
    # 16M bytes is 16MB, which is fine. But computing it takes time.
    # Alternatively, compute on the fly. Comparability check is O(1).
    
    def is_comparable(a, b):
        return (a | b) == a or (a | b) == b

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comparable(family[i], family[j]):
                    cnt += 1
        return cnt

    def get_degrees(family):
        """Return list of degrees for each set in family."""
        degrees = [0] * len(family)
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comparable(family[i], family[j]):
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    # Initial Construction: 
    # Start with all sets of size 0, 1, 11, 12.
    # Then add sets of size 2 and 10 greedily.
    
    # Generate masks by size
    masks_by_size = [[] for _ in range(n+1)]
    for mask in range(1 << n):
        masks_by_size[bin(mask).count('1')].append(mask)
        
    family = []
    seen = set()
    
    # Add size 0 and 12
    family.append(masks_by_size[0][0])
    seen.add(masks_by_size[0][0])
    family.append(masks_by_size[12][0])
    seen.add(masks_by_size[12][0])
    
    # Add size 1 and 11
    for mask in masks_by_size[1]:
        family.append(mask)
        seen.add(mask)
    for mask in masks_by_size[11]:
        family.append(mask)
        seen.add(mask)
        
    # Current size is 1 + 1 + 12 + 12 = 26. Need 14 more.
    # Candidates: size 2 and 10.
    # Calculate degree for each candidate against current family.
    
    candidates = masks_by_size[2] + masks_by_size[10]
    
    # Score each candidate
    cand_scores = []
    for c in candidates:
        if c in seen:
            continue
        deg = 0
        for s in family:
            if is_comparable(c, s):
                deg += 1
        cand_scores.append((deg, c))
        
    # Sort by degree descending
    cand_scores.sort(key=lambda x: -x[0])
    
    # Add top 14
    for deg, c in cand_scores[:14]:
        family.append(c)
        seen.add(c)
        
    # Ensure exactly 40
    if len(family) > 40:
        family = family[:40]
        seen = set(family)
    while len(family) < 40:
        # Find any unused mask
        for mask in range(1 << n):
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                break
                
    # Local Search
    # Objective: Maximize sum of degrees / 2 (which is count of comparable pairs)
    
    current_score = count_comparables(family)
    best_family = list(family)
    best_score = current_score
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        degrees = get_degrees(family)
        
        for i in range(len(family)):
            if time.time() > deadline - 5:
                break
            old_set = family[i]
            old_deg = degrees[i]
            
            # Try to find a better replacement
            # We want to replace old_set with new_set such that:
            # new_deg_in_family_without_old > old_deg
            # Note: new_deg_in_family_without_old is the number of sets in family \ {old_set} comparable to new_set.
            
            best_new = None
            best_gain = 0
            
            # Heuristic: Look at sets close to old_set in Hamming distance? 
            # Or just random sample? 
            # Let's try all sets of size close to old_set's size, or just random.
            # Given 4096 masks, checking all is 4096 * 40 operations per swap.
            # 40 * 4096 * 40 = ~6.5M ops per full pass. This is fast enough in Python if optimized?
            # Maybe a bit slow. Let's sample.
            
            # Strategy: Sample 100 random candidates
            candidates = random.sample(range(1 << n), 100)
            
            for new_set in candidates:
                if new_set == old_set or new_set in seen:
                    continue
                
                # Calculate degree of new_set in family without old_set
                new_deg = 0
                for j, s in enumerate(family):
                    if j == i:
                        continue
                    if is_comparable(new_set, s):
                        new_deg += 1
                
                gain = new_deg - old_deg
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_set
            
            if best_new is not None:
                # Perform swap
                family[i] = best_new
                seen.remove(old_set)
                seen.add(best_new)
                current_score += best_gain
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                    
                # Restart loop to recalculate degrees
                break
                
    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    # Count comparables
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
