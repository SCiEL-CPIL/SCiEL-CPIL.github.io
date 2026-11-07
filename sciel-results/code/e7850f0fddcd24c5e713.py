import time
import random
import itertools

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40

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
        degrees = [0] * len(family)
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comparable(family[i], family[j]):
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    # Generate all masks
    all_masks = list(range(1 << n))
    masks_by_size = [[] for _ in range(n+1)]
    for mask in all_masks:
        masks_by_size[bin(mask).count('1')].append(mask)

    # Chain-based initial construction
    # Create multiple chains that together cover many comparable pairs
    # A chain is a sequence where each set is a subset of the next
    
    # Strategy: Use random chains to build initial family
    def make_random_chain(start_size, end_size, used):
        """Create a random chain from sets of start_size to end_size."""
        chain = []
        current = None
        for sz in range(start_size, end_size + 1):
            if sz == start_size:
                # Pick a random set of this size not in used
                candidates = [m for m in masks_by_size[sz] if m not in used]
                if not candidates:
                    break
                current = random.choice(candidates)
            else:
                # Pick a random superset of current with size sz
                # Current has 'current_bits' bits, need to add (sz - len(current_bits)) bits
                cur_bits = [i for i in range(n) if (current >> i) & 1]
                avail = [i for i in range(n) if i not in cur_bits]
                need = sz - len(cur_bits)
                if need < 0 or need > len(avail):
                    break
                new_bits = random.sample(avail, need)
                new_mask = current
                for b in new_bits:
                    new_mask |= (1 << b)
                if new_mask in used:
                    break
                current = new_mask
            chain.append(current)
            used.add(current)
        return chain

    family = []
    seen = set()
    
    # Build multiple chains
    # Chain 1: empty -> {0} -> {0,1} -> {0,1,2} -> ... -> {0,1,...,11}
    # This gives 13 sets in one chain
    # But we need variety. Let's use several shorter chains.
    
    # Approach: Create 8 chains of length 5 each (40 total)
    # Each chain: size k, k+1, k+2, k+3, k+4 for some starting k
    
    # Better: Mix sizes to maximize cross-chain comparability
    # Use chains that overlap in structure
    
    # Start with a few full chains and fill rest greedily
    # Chain 1: sizes 0,1,2,3,4,5 (6 sets)
    # Chain 2: sizes 6,7,8,9,10,11,12 (7 sets) - but these are supersets of chain 1's end
    # Actually, let's just do greedy with good initial seeds
    
    # Initial: Add empty and full set
    family.append(0)
    seen.add(0)
    family.append((1 << n) - 1)
    seen.add((1 << n) - 1)
    
    # Add all singletons and co-singletons (12 + 12 = 24 sets)
    for mask in masks_by_size[1]:
        family.append(mask)
        seen.add(mask)
    for mask in masks_by_size[11]:
        family.append(mask)
        seen.add(mask)
    
    # Current: 2 + 12 + 12 = 26. Need 14 more.
    # Candidates: size 2, 3, 9, 10
    # Score each candidate by how many existing family members it's comparable to
    
    candidates = []
    for sz in [2, 3, 9, 10]:
        candidates.extend(masks_by_size[sz])
    
    cand_scores = []
    for c in candidates:
        if c in seen:
            continue
        deg = 0
        for s in family:
            if is_comparable(c, s):
                deg += 1
        cand_scores.append((deg, c))
        
    cand_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Add top 14
    for deg, c in cand_scores[:14]:
        family.append(c)
        seen.add(c)
    
    # Ensure exactly 40
    if len(family) > 40:
        family = family[:40]
        seen = set(family)
    while len(family) < 40:
        for mask in all_masks:
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                break
    
    # Local Search with improved candidate generation
    current_score = count_comparables(family)
    best_family = list(family)
    best_score = current_score
    
    def generate_candidates(old_set, family, seen, n):
        """Generate smart candidates for replacement."""
        candidates = set()
        
        # 1. Subsets and supersets of family members (limited)
        for s in family:
            if s == old_set:
                continue
            # If s is subset of old_set, try slightly larger subsets of s
            # If s is superset of old_set, try slightly smaller supersets
            # Just add s's neighbors
            for bit in range(n):
                if (s >> bit) & 1:
                    new_m = s & ~(1 << bit)
                    if new_m != old_set and new_m not in seen:
                        candidates.add(new_m)
                else:
                    new_m = s | (1 << bit)
                    if new_m != old_set and new_m not in seen:
                        candidates.add(new_m)
        
        # 2. Hamming neighbors of old_set
        for bit in range(n):
            new_m = old_set ^ (1 << bit)
            if new_m not in seen:
                candidates.add(new_m)
        
        # 3. Random samples
        for _ in range(50):
            candidates.add(random.choice(all_masks))
        
        # 4. Sets of similar size as old_set
        old_size = bin(old_set).count('1')
        for sz in range(max(0, old_size-2), min(n+1, old_size+3)):
            if sz != old_size:
                for m in random.sample(masks_by_size[sz], min(10, len(masks_by_size[sz]))):
                    if m not in seen and m != old_set:
                        candidates.add(m)
        
        return list(candidates)
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        no_improve_count = 0
        degrees = get_degrees(family)
        
        for i in range(len(family)):
            if time.time() > deadline - 5:
                break
            old_set = family[i]
            old_deg = degrees[i]
            
            best_new = None
            best_gain = 0
            
            candidates = generate_candidates(old_set, family, seen, n)
            
            for new_set in candidates:
                if new_set == old_set or new_set in seen:
                    continue
                
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
                family[i] = best_new
                seen.remove(old_set)
                seen.add(best_new)
                current_score += best_gain
                improved = True
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                
                break
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
    
    # Final polish: try more iterations with smaller perturbations
    if time.time() < deadline - 10:
        polish_family = list(best_family)
        polish_seen = set(polish_family)
        polish_score = best_score
        
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            degrees = get_degrees(polish_family)
            for i in range(len(polish_family)):
                if time.time() > deadline - 5:
                    break
                old_set = polish_family[i]
                old_deg = degrees[i]
                
                best_new = None
                best_gain = 0
                
                # Try all masks of same size and adjacent sizes
                old_size = bin(old_set).count('1')
                for sz in range(max(0, old_size-1), min(n+1, old_size+2)):
                    for new_set in masks_by_size[sz]:
                        if new_set == old_set or new_set in polish_seen:
                            continue
                        
                        new_deg = 0
                        for j, s in enumerate(polish_family):
                            if j == i:
                                continue
                            if is_comparable(new_set, s):
                                new_deg += 1
                        
                        gain = new_deg - old_deg
                        if gain > best_gain:
                            best_gain = gain
                            best_new = new_set
                
                if best_new is not None:
                    polish_family[i] = best_new
                    polish_seen.remove(old_set)
                    polish_seen.add(best_new)
                    polish_score += best_gain
                    
                    if polish_score > best_score:
                        best_score = polish_score
                        best_family = list(polish_family)
    
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
