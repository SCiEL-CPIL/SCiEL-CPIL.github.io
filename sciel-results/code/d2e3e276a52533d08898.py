import numpy as np
from itertools import combinations
import random
import time

def construct():
    # We aim for n=34
    n = 34
    
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute for each vertex, the list of triple indices that contain it
    vertex_triples = [[] for _ in range(n)]
    for i, t in enumerate(all_triples):
        for v in t:
            vertex_triples[v].append(i)
    
    # Precompute for each pair of vertices, the triples containing both
    # This helps for K4/K5 checks
    
    def init_colors(seed=42):
        random.seed(seed)
        color_list = [0] * num_triples
        for i, (a, b, c) in enumerate(all_triples):
            if a < 32 and b < 32 and c < 32:
                # F_2^5: Red if xor = 0
                if (a ^ b ^ c) == 0:
                    color_list[i] = 0
                else:
                    color_list[i] = 1
            else:
                color_list[i] = random.randint(0, 1)
        return color_list
    
    def count_red_in_4(color_list, verts):
        """Count red triples in a 4-set. Return 4 if all red, else < 4."""
        count = 0
        for a, b, c in combinations(verts, 3):
            idx = triple_to_idx[(a, b, c)]
            if color_list[idx] == 0:
                count += 1
            else:
                return count
        return 4
    
    def count_blue_in_5(color_list, verts):
        """Count blue triples in a 5-set. Return 10 if all blue, else < 10."""
        count = 0
        for a, b, c in combinations(verts, 3):
            idx = triple_to_idx[(a, b, c)]
            if color_list[idx] == 1:
                count += 1
            else:
                return count
        return 10
    
    def find_violation(color_list, n_val):
        """Find a Red K4 or Blue K5. Returns (type, vertices) or (None, None)."""
        # Check Red K4s first (fewer of them)
        for verts in combinations(range(n_val), 4):
            if count_red_in_4(color_list, verts) == 4:
                return ('red_k4', verts)
        # Check Blue K5s
        for verts in combinations(range(n_val), 5):
            if count_blue_in_5(color_list, verts) == 10:
                return ('blue_k5', verts)
        return (None, None)
    
    def repair(color_list, n_val, max_iters=200):
        """Attempt to repair violations. Returns modified color_list."""
        for it in range(max_iters):
            vtype, verts = find_violation(color_list, n_val)
            if vtype is None:
                return color_list  # Valid!
            
            if vtype == 'red_k4':
                # Flip one red triple in the K4 to blue
                k4_triples = list(combinations(verts, 3))
                random.shuffle(k4_triples)
                fixed = False
                for t in k4_triples:
                    idx = triple_to_idx[t]
                    old = color_list[idx]
                    color_list[idx] = 1  # Flip to blue
                    # Check if this created a blue K5
                    created_blue_k5 = False
                    t_set = set(t)
                    # Only check 5-sets containing all of t
                    for other1 in range(n_val):
                        if other1 in t_set:
                            continue
                        for other2 in range(other1 + 1, n_val):
                            if other2 in t_set:
                                continue
                            v5 = tuple(sorted(t_set | {other1, other2}))
                            if count_blue_in_5(color_list, v5) == 10:
                                created_blue_k5 = True
                                break
                        if created_blue_k5:
                            break
                    if not created_blue_k5:
                        fixed = True
                        break
                    else:
                        color_list[idx] = old  # Revert
                if not fixed:
                    return color_list  # Stuck
            else:  # blue_k5
                # Flip one blue triple in the K5 to red
                k5_triples = list(combinations(verts, 3))
                random.shuffle(k5_triples)
                fixed = False
                for t in k5_triples:
                    idx = triple_to_idx[t]
                    old = color_list[idx]
                    color_list[idx] = 0  # Flip to red
                    # Check if this created a red K4
                    created_red_k4 = False
                    t_set = set(t)
                    for other in range(n_val):
                        if other in t_set:
                            continue
                        v4 = tuple(sorted(t_set | {other}))
                        if count_red_in_4(color_list, v4) == 4:
                            created_red_k4 = True
                            break
                    if not created_red_k4:
                        fixed = True
                        break
                    else:
                        color_list[idx] = old  # Revert
                if not fixed:
                    return color_list  # Stuck
        
        return color_list
    
    # Try multiple random seeds
    best_n = 0
    best_colors = None
    
    for seed in range(10):
        color_list = init_colors(seed)
        color_list = repair(color_list, n, max_iters=100)
        
        # Verify
        vtype, verts = find_violation(color_list, n)
        if vtype is None:
            best_n = n
            best_colors = color_list
            break
    
    if best_n == n:
        return {'n': best_n, 'colors': best_colors}
    
    # Try smaller n values
    for n_try in range(33, 8, -1):
        all_triples_try = list(combinations(range(n_try), 3))
        triple_to_idx_try = {t: i for i, t in enumerate(all_triples_try)}
        
        def init_colors_try(n_val, seed=42):
            random.seed(seed)
            cl = [0] * len(all_triples_try)
            for i, (a, b, c) in enumerate(all_triples_try):
                if a < 32 and b < 32 and c < 32:
                    if (a ^ b ^ c) == 0:
                        cl[i] = 0
                    else:
                        cl[i] = 1
                else:
                    cl[i] = random.randint(0, 1)
            return cl
        
        def count_red_in_4_try(cl, verts):
            count = 0
            for a, b, c in combinations(verts, 3):
                idx = triple_to_idx_try[(a, b, c)]
                if cl[idx] == 0:
                    count += 1
                else:
                    return count
            return 4
        
        def count_blue_in_5_try(cl, verts):
            count = 0
            for a, b, c in combinations(verts, 3):
                idx = triple_to_idx_try[(a, b, c)]
                if cl[idx] == 1:
                    count += 1
                else:
                    return count
            return 10
        
        def find_violation_try(cl, n_val):
            for verts in combinations(range(n_val), 4):
                if count_red_in_4_try(cl, verts) == 4:
                    return ('red_k4', verts)
            for verts in combinations(range(n_val), 5):
                if count_blue_in_5_try(cl, verts) == 10:
                    return ('blue_k5', verts)
            return (None, None)
        
        def repair_try(cl, n_val, max_iters=100):
            for it in range(max_iters):
                vtype, verts = find_violation_try(cl, n_val)
                if vtype is None:
                    return cl
                if vtype == 'red_k4':
                    k4_triples = list(combinations(verts, 3))
                    random.shuffle(k4_triples)
                    fixed = False
                    for t in k4_triples:
                        idx = triple_to_idx_try[t]
                        old = cl[idx]
                        cl[idx] = 1
                        created = False
                        t_set = set(t)
                        for o1 in range(n_val):
                            if o1 in t_set: continue
                            for o2 in range(o1+1, n_val):
                                if o2 in t_set: continue
                                v5 = tuple(sorted(t_set | {o1, o2}))
                                if count_blue_in_5_try(cl, v5) == 10:
                                    created = True
                                    break
                            if created: break
                        if not created:
                            fixed = True
                            break
                        else:
                            cl[idx] = old
                    if not fixed:
                        return cl
                else:
                    k5_triples = list(combinations(verts, 3))
                    random.shuffle(k5_triples)
                    fixed = False
                    for t in k5_triples:
                        idx = triple_to_idx_try[t]
                        old = cl[idx]
                        cl[idx] = 0
                        created = False
                        t_set = set(t)
                        for o in range(n_val):
                            if o in t_set: continue
                            v4 = tuple(sorted(t_set | {o}))
                            if count_red_in_4_try(cl, v4) == 4:
                                created = True
                                break
                        if not created:
                            fixed = True
                            break
                        else:
                            cl[idx] = old
                    if not fixed:
                        return cl
            return cl
        
        found = False
        for seed in range(10):
            cl = init_colors_try(n_try, seed)
            cl = repair_try(cl, n_try, max_iters=100)
            vtype, verts = find_violation_try(cl, n_try)
            if vtype is None:
                best_n = n_try
                best_colors = cl
                found = True
                break
        if found:
            break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Final fallback: n=1
    return {'n': 1, 'colors': []}
