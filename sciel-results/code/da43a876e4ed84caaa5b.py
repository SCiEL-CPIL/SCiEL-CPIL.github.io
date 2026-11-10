import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 12
    m = 40
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Precompute comparability bitsets
    # comp_mask[m] is a Python int where bit j is 1 if m and j are comparable (strict subset)
    comp_mask = [0] * (1 << n)
    
    # For each mask, find all strict subsets and strict supersets
    # To avoid double work or just do it simply:
    # For each m, iterate subsets s of m (s != m).
    # Then m and s are comparable.
    # Set bit s in comp_mask[m] and bit m in comp_mask[s].
    
    for m in range(1 << n):
        # Iterate strict subsets
        s = (m - 1) & m
        while s:
            comp_mask[m] |= (1 << s)
            comp_mask[s] |= (1 << m)
            s = (s - 1) & m
        
        # Iterate strict supersets?
        # Instead of iterating supersets for each m, we can just rely on the subset iteration?
        # If we iterate subsets for ALL m, then for any pair (a, b) with a < b,
        # when m=b, we find a. We set bit a in comp_mask[b] and bit b in comp_mask[a].
        # This covers all comparable pairs exactly once in the loop structure.
        # So we don't need to iterate supersets separately.
        
    # Wait, the loop above:
    # for m in range(1 << n):
    #   s = (m - 1) & m
    #   while s:
    #     ...
    #     s = (s - 1) & m
    # This iterates all proper subsets s of m.
    # So for any pair {a, b} where a is a strict subset of b, the pair is processed when m=b and s=a.
    # We set comp_mask[b] bit a and comp_mask[a] bit b.
    # This is correct and efficient.
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(len(family)):
            mi = family[i]
            for j in range(i + 1, len(family)):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count

    def score_candidate_bitset(cand, family_bitset):
        """Count how many sets in family (represented by bitset) are comparable with cand."""
        # family_bitset has bits set for all masks in the family.
        # We want count of bits set in (family_bitset & comp_mask[cand]).
        # Note: comp_mask[cand] does NOT have bit cand set (strict subset).
        # So if cand is in family_bitset, it won't count itself, which is correct.
        return bin(family_bitset & comp_mask[cand]).count('1')
    
    def initial_structure_pure_chain():
        family = []
        used = set()
        chain_mask = 0
        for k in range(n + 1):
            family.append(chain_mask)
            used.add(chain_mask)
            if k < n:
                chain_mask |= (1 << k)
        
        remaining = [mm for mm in range(1 << n) if mm not in used]
        random.shuffle(remaining)
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            # Greedy pick: pick one that maximizes comparability with current family
            # But for initialization, random might be fine, or just take first few.
            # Let's just take first few from remaining for speed.
            best = remaining.pop(0)
            family.append(best)
            used.add(best)
        
        return family

    def initial_structure_ball():
        """
        Initialize with a 'ball' (down-set like structure).
        Include all sets of size 0 and 1.
        Randomly pick rest from size 2.
        """
        family = []
        used = set()
        
        # Size 0
        family.append(0)
        used.add(0)
        
        # Size 1
        for i in range(n):
            mask = 1 << i
            family.append(mask)
            used.add(mask)
            
        # Need m - 13 more sets
        # Pick random sets of size 2
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if len(family) >= m:
                break
            mask = (1 << i) | (1 << j)
            if mask not in used:
                family.append(mask)
                used.add(mask)
                
        # If still not enough (should not happen as there are 66 pairs), fill with random
        if len(family) < m:
            remaining = [mm for mm in range(1 << n) if mm not in used]
            random.shuffle(remaining)
            for mm in remaining:
                if len(family) >= m:
                    break
                family.append(mm)
                used.add(mm)
        
        return family

    def local_search_fast(family, time_limit):
        """Fast local search using bitsets."""
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        
        # Build initial bitset
        family_bitset = 0
        for mask in family:
            family_bitset |= (1 << mask)
            
        used = set(family)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            for i in range(len(family)):
                if time.time() > end_time:
                    return family
                
                # Remove family[i] from bitset
                old_mask = family[i]
                family_bitset ^= (1 << old_mask)
                
                best_gain = 0
                best_cand = None
                
                # Try all candidates not in family
                # We can iterate all 4096 masks
                for cand in range(1 << n):
                    if cand in used:
                        continue
                    # Score cand with current family (without old_mask)
                    # Note: score_candidate_bitset counts comparability with family_bitset.
                    # family_bitset does NOT contain old_mask.
                    # It DOES contain other family members.
                    # This is the gain from adding cand to the rest.
                    # But we are replacing old_mask.
                    # Gain = Score(cand, Rest) - Score(old_mask, Rest)
                    # We don't have Score(old_mask, Rest) easily.
                    # However, we can compute Score(cand, Rest) and compare to a baseline?
                    # No, we need the actual gain.
                    # Let's compute Score(old_mask, Rest) once.
                    # Score(old_mask, Rest) is fixed for this i.
                    # But we haven't computed it.
                    # We can compute it before the loop.
                    
                    # Actually, let's just compute the new total score?
                    # No, that's expensive.
                    
                    # Let's precompute score_old = score_candidate_bitset(old_mask, family_bitset)
                    # But family_bitset is changing? No, for this i, it's fixed.
                    # So:
                    # score_old = bin(family_bitset & comp_mask[old_mask]).count('1')
                    # Then for each cand:
                    # score_cand = bin(family_bitset & comp_mask[cand]).count('1')
                    # gain = score_cand - score_old
                    
                    # We can compute score_old once.
                    # Let's move this logic.
                    pass
                
                # Optimization: Compute score_old once
                score_old = bin(family_bitset & comp_mask[old_mask]).count('1')
                
                for cand in range(1 << n):
                    if cand in used:
                        continue
                    score_cand = bin(family_bitset & comp_mask[cand]).count('1')
                    gain = score_cand - score_old
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None and best_gain > 0:
                    family[i] = best_cand
                    current_score += best_gain
                    improved = True
                    # Update bitset
                    family_bitset ^= (1 << best_cand)
                    used.remove(old_mask)
                    used.add(best_cand)
                else:
                    # No improvement, put old_mask back
                    family_bitset ^= (1 << old_mask)
                    # used was not changed, so no need to update used
                    pass
        
        return family

    # Try multiple initial structures
    best_family = None
    best_score = -1
    
    # 1. Chain
    fam1 = initial_structure_pure_chain()
    if len(fam1) == m:
        s1 = count_comparable(fam1)
        if s1 > best_score:
            best_score = s1
            best_family = fam1[:]
            
    # 2. Ball
    if time.time() < deadline - 5:
        fam2 = initial_structure_ball()
        if len(fam2) == m:
            s2 = count_comparable(fam2)
            if s2 > best_score:
                best_score = s2
                best_family = fam2[:]
                
    # 3. Random
    if time.time() < deadline - 5:
        fam3 = random.sample(range(1 << n), m)
        s3 = count_comparable(fam3)
        if s3 > best_score:
            best_score = s3
            best_family = fam3[:]

    # Local search on best
    if best_family and time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        best_family = local_search_fast(best_family, time_remaining)
        final_score = count_comparable(best_family)
        if final_score > best_score:
            best_score = final_score
            
    # If time permits, try local search on other initializations?
    # The local search is fast now, so we can afford to run it on all 3 and pick best.
    
    # Let's restart the logic to run local search on all candidates and pick the global best.
    
    candidates = []
    
    fam1 = initial_structure_pure_chain()
    if len(fam1) == m:
        candidates.append(fam1)
        
    fam2 = initial_structure_ball()
    if len(fam2) == m:
        candidates.append(fam2)
        
    fam3 = random.sample(range(1 << n), m)
    candidates.append(fam3)
    
    # Run local search on each candidate if time permits
    # We have ~80s. Precomp is fast.
    # Local search might take a few seconds per run.
    # Let's run on the top 2 initial scores first.
    
    scored_candidates = []
    for fam in candidates:
        s = count_comparable(fam)
        scored_candidates.append((s, fam))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_family = scored_candidates[0][1]
    best_score = scored_candidates[0][0]
    
    # Run local search on the best candidate
    if time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        # Split time among top candidates?
        # Let's just run on the best one for now.
        improved_family = local_search_fast(best_family, time_remaining)
        improved_score = count_comparable(improved_family)
        if improved_score > best_score:
            best_family = improved_family
            best_score = improved_score
            
    # If time still left, run on second best?
    if len(scored_candidates) > 1 and time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        fam2 = scored_candidates[1][1]
        improved_family2 = local_search_fast(fam2, time_remaining)
        improved_score2 = count_comparable(improved_family2)
        if improved_score2 > best_score:
            best_family = improved_family2
            best_score = improved_score2

    # Ensure exactly m sets
    if best_family is None or len(best_family) != m:
        best_family = random.sample(range(1 << n), m)
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    # Validate
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
        assert len(s) == len(set(s))
        
    return result
