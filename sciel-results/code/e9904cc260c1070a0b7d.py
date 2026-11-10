import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    n = 12
    m = 40
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
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
    
    def score_candidate(cand, family):
        """Count how many sets in family are comparable with cand."""
        s = 0
        for fs in family:
            if (cand & fs) == cand and cand != fs:
                s += 1
            elif (cand & fs) == fs and cand != fs:
                s += 1
        return s
    
    # Generate all 2^12 = 4096 subsets as bitmasks
    all_masks = list(range(1 << n))
    
    def initial_structure_pure_chain():
        """Start with a chain, then greedily add."""
        family = []
        used = set()
        
        chain_mask = 0
        for k in range(n + 1):
            family.append(chain_mask)
            used.add(chain_mask)
            if k < n:
                chain_mask |= (1 << k)
        
        remaining = [mm for mm in all_masks if mm not in used]
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                if cand in used:
                    continue
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def initial_structure_chain_star():
        """Chain + star hybrid."""
        family = []
        used = set()
        
        family.append(0)
        used.add(0)
        
        for i in range(n):
            mask = 1 << i
            family.append(mask)
            used.add(mask)
        
        remaining = [mm for mm in all_masks if mm not in used]
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                if cand in used:
                    continue
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def local_search_fast(family, time_limit):
        """Fast local search using incremental gain computation."""
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used = set(family)
            
            for i in range(len(family)):
                if time.time() > end_time:
                    return family
                
                # Compute comparability of family[i] with rest of family
                rest = family[:i] + family[i+1:]
                score_i_with_rest = score_candidate(family[i], rest)
                
                best_replacement = None
                best_gain = 0
                
                for cand in all_masks:
                    if cand in used:
                        continue
                    score_cand_with_rest = score_candidate(cand, rest)
                    gain = score_cand_with_rest - score_i_with_rest
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = cand
                
                if best_replacement is not None and best_gain > 0:
                    family[i] = best_replacement
                    current_score += best_gain
                    improved = True
                    used = set(family)
        
        return family
    
    def two_swap_local_search(family, time_limit):
        """Try swapping two sets simultaneously to escape local optima."""
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used = set(family)
            
            for i in range(len(family)):
                if time.time() > end_time:
                    return family
                for j in range(i + 1, len(family)):
                    if time.time() > end_time:
                        return family
                    
                    # Compute comparability of family[i] and family[j] with rest
                    rest = family[:i] + family[i+1:j] + family[j+1:]
                    score_i_with_rest = score_candidate(family[i], rest)
                    score_j_with_rest = score_candidate(family[j], rest)
                    # Also need comparability between i and j
                    mi, mj = family[i], family[j]
                    if (mi & mj) == mi and mi != mj:
                        ij_comp = 1
                    elif (mj & mi) == mj and mj != mi:
                        ij_comp = 1
                    else:
                        ij_comp = 0
                    
                    best_c1, best_c2 = None, None
                    best_gain = 0
                    
                    # Try all pairs of candidates
                    # This is expensive, so limit to promising candidates
                    # First, find top candidates for each position
                    cands_for_i = []
                    cands_for_j = []
                    for cand in all_masks:
                        if cand in used:
                            continue
                        s_i = score_candidate(cand, rest)
                        s_j = score_candidate(cand, rest)
                        cands_for_i.append((s_i, cand))
                        cands_for_j.append((s_j, cand))
                    
                    cands_for_i.sort(reverse=True)
                    cands_for_j.sort(reverse=True)
                    
                    # Try top 50 for each
                    for s1, c1 in cands_for_i[:50]:
                        if c1 == family[j]:
                            continue
                        for s2, c2 in cands_for_j[:50]:
                            if c2 == family[i] or c2 == c1:
                                continue
                            # Compute comparability between c1 and c2
                            if (c1 & c2) == c1 and c1 != c2:
                                c12_comp = 1
                            elif (c2 & c1) == c2 and c2 != c1:
                                c12_comp = 1
                            else:
                                c12_comp = 0
                            
                            new_score = current_score - score_i_with_rest - score_j_with_rest - ij_comp + s1 + s2 + c12_comp
                            gain = new_score - current_score
                            if gain > best_gain:
                                best_gain = gain
                                best_c1, best_c2 = c1, c2
                    
                    if best_c1 is not None and best_gain > 0:
                        family[i] = best_c1
                        family[j] = best_c2
                        current_score += best_gain
                        improved = True
                        used = set(family)
                        break
                if improved:
                    break
        
        return family
    
    # Try multiple initial structures and pick the best
    best_family = None
    best_score = -1
    
    fam1 = initial_structure_pure_chain()
    if len(fam1) == m:
        s1 = count_comparable(fam1)
        if s1 > best_score:
            best_score = s1
            best_family = fam1[:]
    
    if time.time() < deadline - 10:
        fam2 = initial_structure_chain_star()
        if len(fam2) == m:
            s2 = count_comparable(fam2)
            if s2 > best_score:
                best_score = s2
                best_family = fam2[:]
    
    # Local search on the best family
    if best_family and time.time() < deadline - 3:
        time_remaining = deadline - 3 - time.time()
        best_family = local_search_fast(best_family, time_remaining)
        final_score = count_comparable(best_family)
        if final_score > best_score:
            best_score = final_score
    
    # Two-swap local search if time permits
    if best_family and time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        best_family = two_swap_local_search(best_family, time_remaining)
        final_score = count_comparable(best_family)
        if final_score > best_score:
            best_score = final_score
    
    # Ensure exactly m sets
    if best_family is None or len(best_family) != m:
        best_family = sorted(all_masks, key=lambda x: (bin(x).count('1'), x))[:m]
    
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
