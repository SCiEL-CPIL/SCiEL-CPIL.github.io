"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i+1, m):
            sj = family[j]
            if (si & sj) == si:
                count += 1
            elif (si & sj) == sj:
                count += 1
    return count

def _greedy_chain_init(n, m):
    """Initialize by building chains: pick a chain of size k, then another disjoint-ish chain, etc."""
    family = []
    used = set()
    max_mask = 1 << n
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        candidates = list(range(max_mask))
        random.shuffle(candidates)
        candidates_to_check = candidates[:500]
        if 0 not in candidates_to_check:
            candidates_to_check.append(0)
        if (max_mask - 1) not in candidates_to_check:
            candidates_to_check.append(max_mask - 1)
        
        for s in candidates_to_check:
            if s in used:
                continue
            
            gain = 0
            for t in family:
                if (s & t) == s:
                    gain += 1
                elif (t & s) == t:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s

        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _two_opt_local_search(family, n, m, deadline):
    """Vectorized 2-opt: try replacing each set with every other possible set using numpy."""
    max_mask = 1 << n
    current_family = np.array(family, dtype=np.int32)
    current_fs = set(current_family.tolist())
    current_score = _calc_score_bitmask(current_family.tolist())
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute all masks as numpy array
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        # Try each position
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_set = current_family[idx]
            
            # Vectorized evaluation of all 4096 candidates
            others = np.delete(current_family, idx)  # shape (m-1,)
            
            # For each candidate new_set, compute:
            # new_comparable = sum over others of [new_set subset other OR other subset new_set]
            # Using numpy broadcasting: (39, 4096)
            
            # new_set & other == new_set means new_set is subset of other
            # other & new_set == other means other is subset of new_set
            
            # We need to check strict subset (not equal)
            # Since all sets in family are distinct, and new_set != old_set,
            # but new_set could equal some other set in the family
            
            # Compute for all candidates at once
            # candidates shape: (4096,)
            # others shape: (39,)
            
            # (4096, 39) broadcast
            cand = all_masks[:, np.newaxis]  # (4096, 1)
            oth = others[np.newaxis, :]       # (1, 39)
            
            # new_set subset of other: (cand & oth) == cand
            cand_subset_oth = (cand & oth) == cand  # (4096, 39)
            
            # other subset of new_set: (oth & cand) == oth
            oth_subset_cand = (oth & cand) == oth   # (4096, 39)
            
            # But we need to exclude equality (strict subset only)
            # cand == oth means they're the same set
            eq_mask = cand == oth  # (4096, 39)
            
            # Strict: cand_subset_oth AND NOT eq
            strict_cand_subset = cand_subset_oth & ~eq_mask
            strict_oth_subset = oth_subset_cand & ~eq_mask
            
            # Count comparabilities for each candidate
            new_comps = strict_cand_subset.sum(axis=1) + strict_oth_subset.sum(axis=1)  # (4096,)
            
            # Old comparabilities for old_set with others
            old_comps = 0
            for other in others:
                if (old_set & other) == old_set and old_set != other:
                    old_comps += 1
                elif (old_set & other) == other and old_set != other:
                    old_comps += 1
            
            # Delta for each candidate
            deltas = new_comps - old_comps  # (4096,)
            
            # Mask out invalid candidates (already in family or same as old_set)
            valid_mask = np.ones(max_mask, dtype=bool)
            for s in current_fs:
                valid_mask[s] = False
            valid_mask[old_set] = False  # don't replace with itself
            
            deltas[~valid_mask] = -999999
            
            # Find best
            max_idx = np.argmax(deltas)
            max_delta = deltas[max_idx]
            
            if max_delta > best_delta:
                best_delta = int(max_delta)
                best_idx = idx
                best_new_set = int(all_masks[max_idx])
        
        if best_delta > 0 and best_idx >= 0:
            # Apply the best move
            old_set = int(current_family[best_idx])
            current_fs.remove(old_set)
            current_fs.add(best_new_set)
            current_family[best_idx] = best_new_set
            current_score += best_delta
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family.tolist()
        else:
            no_improvement_count += 1
            # If stuck, do a random perturbation
            if no_improvement_count >= 3:
                # Try several random perturbations
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    new_set = random.randint(0, max_mask - 1)
                    if new_set in current_fs:
                        continue
                    old_set = int(current_family[idx])
                    
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        other = int(current_family[j])
                        if (old_set & other) == old_set and old_set != other:
                            delta -= 1
                        elif (old_set & other) == other and old_set != other:
                            delta -= 1
                        if (new_set & other) == new_set and new_set != other:
                            delta += 1
                        elif (new_set & other) == other and new_set != old_set:
                            delta += 1
                    
                    if delta >= 0 or random.random() < 0.3:
                        current_fs.remove(old_set)
                        current_fs.add(new_set)
                        current_family[idx] = new_set
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_family = current_family.tolist()
                        break
                no_improvement_count = 0
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Initial family
    current_family = _greedy_chain_init(n, m)
    
    # 2-opt local search
    best_family, best_score = _two_opt_local_search(current_family, n, m, deadline)
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
