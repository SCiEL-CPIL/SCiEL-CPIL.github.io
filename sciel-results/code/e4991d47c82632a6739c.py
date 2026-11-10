# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    forbidden_set = set(forbidden)
    
    # Precompute conflict masks using bitmasks for speed
    # conflict_mask[i] is a 205-bit integer where bit j is 1 if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                conflict_mask[i] |= (1 << j)
    
    def check_conflict(a, b):
        return (conflict_mask[a] >> b) & 1
    
    def greedy_build(order):
        selected = []
        for cand in order:
            ok = True
            for s in selected:
                if check_conflict(cand, s):
                    ok = False
                    break
            if ok:
                selected.append(cand)
        return selected
    
    def try_add(s, current):
        """Try to add element s to current set"""
        mask_s = conflict_mask[s]
        for e in current:
            if (mask_s >> e) & 1:
                return False
        return True
    
    def find_conflicts(s, current):
        """Find all elements in current that conflict with s"""
        mask_s = conflict_mask[s]
        confs = []
        for e in current:
            if (mask_s >> e) & 1:
                confs.append(e)
        return confs
    
    def local_search(current, max_time):
        """Enhanced local search with systematic swap strategies using bitmask acceleration"""
        current = list(current)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            current_set = set(current)
            unselected = [i for i in range(m) if i not in current_set]
            
            # Strategy 1: Add elements with 0 conflicts
            for s in unselected:
                if time.time() >= max_time:
                    break
                if not find_conflicts(s, current):
                    current.append(s)
                    improved = True
            
            # Strategy 2: Single swaps (remove 1, add 1)
            if time.time() < max_time:
                current_set = set(current)
                unselected = [i for i in range(m) if i not in current_set]
                for s in unselected:
                    if time.time() >= max_time:
                        break
                    confs = find_conflicts(s, current)
                    if len(confs) == 1:
                        c = confs[0]
                        current.remove(c)
                        current.append(s)
                        improved = True
            
            # Strategy 3: Double swaps (remove 2, add 2)
            if time.time() < max_time:
                current_set = set(current)
                unselected = [i for i in range(m) if i not in current_set]
                
                # Limit search space for performance
                if len(unselected) > 50:
                    rng_local = random.Random(hash(tuple(sorted(current))) % 10000)
                    unselected = rng_local.sample(unselected, 50)
                
                for s1 in unselected:
                    if time.time() >= max_time:
                        break
                    confs1 = find_conflicts(s1, current)
                    if len(confs1) > 3:
                        continue
                    
                    for s2 in unselected:
                        if s2 == s1 or time.time() >= max_time:
                            continue
                        confs2 = find_conflicts(s2, current)
                        if len(confs2) > 3:
                            continue
                        
                        # Check if s1 and s2 conflict with each other
                        if check_conflict(s1, s2):
                            continue
                        
                        to_remove = set(confs1) | set(confs2)
                        if len(to_remove) > 2:
                            continue
                        
                        # Check if s1 and s2 conflict with anyone not in to_remove
                        ok = True
                        for e in current:
                            if e in to_remove:
                                continue
                            if check_conflict(s1, e) or check_conflict(s2, e):
                                ok = False
                                break
                        if ok:
                            for e in to_remove:
                                current.remove(e)
                            current.append(s1)
                            current.append(s2)
                            improved = True
            
            # Strategy 4: Triple swaps (remove 3, add 3)
            if time.time() < max_time:
                current_set = set(current)
                unselected = [i for i in range(m) if i not in current_set]
                
                # Limit search space
                if len(unselected) > 30:
                    rng_local = random.Random(hash(tuple(sorted(current))) % 10000 + 1)
                    unselected = rng_local.sample(unselected, 30)
                
                for s1 in unselected[:20]:
                    if time.time() >= max_time:
                        break
                    confs1 = find_conflicts(s1, current)
                    if len(confs1) > 4:
                        continue
                    
                    for s2 in unselected[:20]:
                        if s2 == s1 or time.time() >= max_time:
                            continue
                        if check_conflict(s1, s2):
                            continue
                        confs2 = find_conflicts(s2, current)
                        if len(confs2) > 4:
                            continue
                        
                        for s3 in unselected[:20]:
                            if s3 == s1 or s3 == s2 or time.time() >= max_time:
                                continue
                            if check_conflict(s1, s3) or check_conflict(s2, s3):
                                continue
                            confs3 = find_conflicts(s3, current)
                            if len(confs3) > 4:
                                continue
                            
                            to_remove = set(confs1) | set(confs2) | set(confs3)
                            if len(to_remove) > 3:
                                continue
                            
                            # Check if s1, s2, s3 conflict with anyone not in to_remove
                            ok = True
                            for e in current:
                                if e in to_remove:
                                    continue
                                if check_conflict(s1, e) or check_conflict(s2, e) or check_conflict(s3, e):
                                    ok = False
                                    break
                            if ok:
                                for e in to_remove:
                                    current.remove(e)
                                current.extend([s1, s2, s3])
                                improved = True
            
            # Strategy 5: Quadruple swaps (remove 4, add 4)
            if time.time() < max_time:
                current_set = set(current)
                unselected = [i for i in range(m) if i not in current_set]
                
                # Limit search space
                if len(unselected) > 25:
                    rng_local = random.Random(hash(tuple(sorted(current))) % 10000 + 2)
                    unselected = rng_local.sample(unselected, 25)
                
                for s1 in unselected[:15]:
                    if time.time() >= max_time:
                        break
                    confs1 = find_conflicts(s1, current)
                    if len(confs1) > 5:
                        continue
                    
                    for s2 in unselected[:15]:
                        if s2 == s1 or time.time() >= max_time:
                            continue
                        if check_conflict(s1, s2):
                            continue
                        confs2 = find_conflicts(s2, current)
                        if len(confs2) > 5:
                            continue
                        
                        for s3 in unselected[:15]:
                            if s3 == s1 or s3 == s2 or time.time() >= max_time:
                                continue
                            if check_conflict(s1, s3) or check_conflict(s2, s3):
                                continue
                            confs3 = find_conflicts(s3, current)
                            if len(confs3) > 5:
                                continue
                            
                            for s4 in unselected[:15]:
                                if s4 == s1 or s4 == s2 or s4 == s3 or time.time() >= max_time:
                                    continue
                                if check_conflict(s1, s4) or check_conflict(s2, s4) or check_conflict(s3, s4):
                                    continue
                                confs4 = find_conflicts(s4, current)
                                if len(confs4) > 5:
                                    continue
                                
                                to_remove = set(confs1) | set(confs2) | set(confs3) | set(confs4)
                                if len(to_remove) > 4:
                                    continue
                                
                                # Check if s1, s2, s3, s4 conflict with anyone not in to_remove
                                ok = True
                                for e in current:
                                    if e in to_remove:
                                        continue
                                    if check_conflict(s1, e) or check_conflict(s2, e) or check_conflict(s3, e) or check_conflict(s4, e):
                                        ok = False
                                        break
                                if ok:
                                    for e in to_remove:
                                        current.remove(e)
                                    current.extend([s1, s2, s3, s4])
                                    improved = True
            
            # Strategy 6: Random restarts within local search
            if time.time() < max_time and not improved:
                # Try random perturbations
                for _ in range(15):
                    if time.time() >= max_time:
                        break
                    perturbed = list(current)
                    num_remove = random.randint(1, min(5, len(perturbed)))
                    to_remove = random.sample(perturbed, num_remove)
                    for e in to_remove:
                        perturbed.remove(e)
                    
                    # Greedily add back
                    current_set = set(perturbed)
                    candidates = to_remove + [i for i in range(m) if i not in current_set]
                    random.shuffle(candidates)
                    
                    for cand in candidates:
                        if time.time() >= max_time:
                            break
                        if cand not in current_set and try_add(cand, perturbed):
                            perturbed.append(cand)
                            current_set.add(cand)
                    
                    if len(perturbed) > len(current):
                        current = perturbed
                        improved = True
                        break
        
        return current
    
    # Try multiple greedy orders to get diverse starting points
    best = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Try natural order
    sel = greedy_build(range(m))
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Try reversed
    if time.time() < deadline:
        sel = greedy_build(range(m-1, -1, -1))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Try random orders
    for _ in range(50):
        if time.time() >= deadline - 80:
            break
        order = list(range(m))
        rng.shuffle(order)
        sel = greedy_build(order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Run local search on the best greedy result
    if time.time() < deadline - 40:
        current = local_search(best, deadline - 15)
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Try local search on a few other good greedy results
    if time.time() < deadline - 40:
        # Get top 5 different greedy results
        greedy_results = []
        seen = set()
        for seed in range(30):
            if time.time() >= deadline - 40:
                break
            order = list(range(m))
            random.Random(seed).shuffle(order)
            sel = greedy_build(order)
            key = tuple(sorted(sel))
            if key not in seen:
                seen.add(key)
                greedy_results.append(sel)
        
        # Sort by size and try local search on top few
        greedy_results.sort(key=len, reverse=True)
        for gr in greedy_results[:5]:
            if time.time() >= deadline - 15:
                break
            if len(gr) + 5 <= best_size:  # Not promising
                continue
            current = local_search(gr, deadline - 10)
            if len(current) > best_size:
                best = current
                best_size = len(current)
    
    # Final verification
    result = sorted(set(best))
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if check_conflict(result[i], result[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to greedy if something went wrong
        result = greedy_build(range(m))
    
    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
