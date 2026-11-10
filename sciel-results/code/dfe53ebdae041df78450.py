"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 12, 'm': 40}

def _count_comparable(family_masks):
    """Count comparable pairs in a family of bitmask integers."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family_masks[i], family_masks[j]
            if a & b == a:  # a is subset of b
                count += 1
            elif a & b == b:  # b is subset of a
                count += 1
    return count

def _greedy_construct(n_vertices, m, deadline):
    """Greedily construct a family of m subsets of {0..n_vertices-1} maximizing comparable pairs."""
    # All possible subsets as bitmasks
    all_masks = list(range(1 << n_vertices))
    all_masks_set = set(all_masks)
    
    family = []
    family_set = set()
    
    # Start with empty set (maximally comparable with everything)
    family.append(0)
    family_set.add(0)
    
    # Precompute subset/superset relationships for speed
    # For each mask, precompute which other masks are comparable
    # But that's 64*64 = 4096 entries, fine
    
    while len(family) < m:
        best_mask = -1
        best_gain = -1
        best_score = -1
        
        for mask in all_masks:
            if mask in family_set:
                continue
            
            # Count how many existing family members are comparable with this mask
            gain = 0
            for fam_mask in family:
                if (mask & fam_mask) == mask or (mask & fam_mask) == fam_mask:
                    gain += 1
            
            if gain > best_gain or (gain == best_gain and mask < best_mask):
                best_gain = gain
                best_mask = mask
                best_score = gain
        
        if best_mask < 0:
            break
        
        family.append(best_mask)
        family_set.add(best_mask)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return family

def construct():
    deadline = time.time() + 45  # Safety margin before 60s limit
    
    # Try with 6 vertices first (64 subsets, need 40)
    # This is the most promising: dense poset
    best_family = None
    best_score = -1
    
    # Greedy with 6 vertices
    family = _greedy_construct(6, 40, deadline)
    if len(family) == 40:
        score = _count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = family
    
    # Also try a different greedy order: start with empty, then size 1, etc.
    # Alternative: greedy but prioritize by size to build layers
    if time.time() < deadline - 5:
        # Layered greedy: pick sets level by level
        all_masks_by_size = {}
        for mask in range(64):
            sz = bin(mask).count('1')
            if sz not in all_masks_by_size:
                all_masks_by_size[sz] = []
            all_masks_by_size[sz].append(mask)
        
        family2 = [0]
        family2_set = {0}
        
        # For each size from 1 to 6, greedily pick best sets
        for sz in range(1, 7):
            candidates = [m for m in all_masks_by_size.get(sz, []) if m not in family2_set]
            # Sort candidates by how many comparable pairs they'd add
            scored = []
            for m in candidates:
                gain = 0
                for fm in family2:
                    if (m & fm) == m or (m & fm) == fm:
                        gain += 1
                scored.append((gain, m))
            scored.sort(key=lambda x: (-x[0], x[1]))
            
            # Add as many as needed up to 40 total
            while len(family2) < 40 and scored:
                gain, m = scored.pop(0)
                family2.append(m)
                family2_set.add(m)
            
            if len(family2) >= 40:
                break
        
        if len(family2) == 40:
            score2 = _count_comparable(family2)
            if score2 > best_score:
                best_score = score2
                best_family = family2
    
    # Convert to lists of vertex indices
    if best_family is None:
        # Fallback: first 40 masks
        best_family = list(range(40))
    
    result = []
    for mask in best_family:
        result.append([b for b in range(6) if (mask >> b) & 1])
    
    # Verify: exactly 40, distinct, each is list of distinct ints
    assert len(result) == 40
    assert len(set(tuple(s) for s in result)) == 40
    for s in result:
        assert isinstance(s, list)
        assert len(s) == len(set(s))
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool)
            assert 0 <= x <= 11
    
    return result
# EVOLVE-BLOCK-END
