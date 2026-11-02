import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors as a numpy array
    neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    rng = np.random.RandomState(42)
    
    # Precompute a "ball" matrix for vectorized operations
    # ball[cw, j] = neighbors[cw, j]
    # We'll use this heavily
    
    while time.time() < deadline - 10.0:
        # === Greedy Construction ===
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline - 15.0:
                break
            
            # Vectorized: count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            
            if len(candidates) > 1:
                # Secondary: prefer candidates with lower total overlap (spread out)
                total_cov = cover_count[neighbors[candidates]].sum(axis=1)
                min_total = np.min(total_cov)
                valid = candidates[total_cov == min_total]
                if len(valid) > 0:
                    candidates = valid
            
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(int(best_idx))
            
            ball = neighbors[best_idx]
            num_newly = np.sum(uncovered_mask[ball])
            uncovered_count -= num_newly
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        if uncovered_count > 0:
            # Try to complete with a simpler greedy
            while uncovered_count > 0 and time.time() < deadline - 15.0:
                U = np.where(uncovered_mask)[0]
                if len(U) == 0:
                    break
                # Find a word that covers the most uncovered
                counts = uncovered_mask[neighbors].sum(axis=1)
                best_idx = np.argmax(counts)
                if counts[best_idx] == 0:
                    break
                if best_idx not in selected:
                    selected.append(int(best_idx))
                ball = neighbors[best_idx]
                num_newly = np.sum(uncovered_mask[ball])
                uncovered_count -= num_newly
                cover_count[ball] += 1
                uncovered_mask[ball] = False
        
        if uncovered_count > 0:
            continue
        
        # === Local Search ===
        cur_size = len(selected)
        
        # Phase 1: Aggressive removal
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            order = list(range(len(selected)))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 12.0:
                    break
                cw = selected[i]
                if np.min(cover_count[neighbors[cw]]) > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
        
        # Phase 2: 1-swap local search (vectorized)
        swap_changed = True
        while swap_changed and time.time() < deadline - 10.0:
            swap_changed = False
            for i in range(len(selected)):
                if time.time() > deadline - 12.0:
                    break
                cw_old = selected[i]
                # Remove cw_old
                cover_count[neighbors[cw_old]] -= 1
                U = np.where(cover_count == 0)[0]
                
                if len(U) == 0:
                    # Redundant - just remove
                    selected.pop(i)
                    swap_changed = True
                    break
                elif len(U) <= 3:
                    # Find candidates that cover all of U
                    cand_set = neighbor_sets[U[0]]
                    for u in U[1:]:
                        cand_set &= neighbor_sets[u]
                    
                    if cand_set:
                        sel_set = set(selected)
                        valid = [c for c in cand_set if c not in sel_set]
                        if valid:
                            # Pick the one with max additional coverage
                            best_c = max(valid, key=lambda c: np.sum(cover_count[neighbors[c]] == 0))
                            selected[i] = best_c
                            cover_count[neighbors[best_c]] += 1
                            swap_changed = True
                            break
                
                # Restore
                cover_count[neighbors[cw_old]] += 1
        
        # Phase 3: 2-exchange
        if len(selected) > 490 and time.time() < deadline - 15.0:
            ex_changed = True
            while ex_changed and time.time() < deadline - 12.0:
                ex_changed = False
                if len(selected) < 2:
                    break
                i1 = rng.randint(len(selected))
                i2 = rng.randint(len(selected))
                if i1 == i2:
                    continue
                
                cw1 = selected[i1]
                cw2 = selected[i2]
                
                cover_count[neighbors[cw1]] -= 1
                cover_count[neighbors[cw2]] -= 1
                
                U = np.where(cover_count == 0)[0]
                
                if len(U) == 0:
                    # Both redundant
                    if i1 > i2:
                        i1, i2 = i2, i1
                    selected.pop(i1)
                    selected.pop(i2)
                    ex_changed = True
                    break
                elif len(U) <= 4:
                    # Find two candidates that cover all of U
                    cand_set = neighbor_sets[U[0]]
                    for u in U[1:]:
                        cand_set &= neighbor_sets[u]
                    
                    if cand_set:
                        sel_set = set(selected)
                        valid = [c for c in cand_set if c not in sel_set]
                        
                        if len(valid) >= 2:
                            # Try pairs
                            found = False
                            # Score candidates
                            scored = sorted(valid, key=lambda c: -np.sum(cover_count[neighbors[c]] == 0))
                            scored = scored[:30]
                            
                            for a in scored:
                                a_covers = set(neighbors[a].tolist())
                                remaining = [u for u in U if u not in a_covers]
                                if len(remaining) == 0:
                                    # a covers all, need b to cover nothing extra (any valid b)
                                    b = scored[0] if scored[0] != a else (scored[1] if len(scored) > 1 else None)
                                    if b is not None:
                                        if i1 > i2:
                                            i1, i2 = i2, i1
                                        selected[i1] = a
                                        selected[i2] = b
                                        cover_count[neighbors[a]] += 1
                                        cover_count[neighbors[b]] += 1
                                        ex_changed = True
                                        found = True
                                        break
                                elif len(remaining) == 1:
                                    u_rem = remaining[0]
                                    b_cands = [b for b in scored if b != a and u_rem in neighbor_sets[b]]
                                    if b_cands:
                                        b = b_cands[0]
                                        # Verify
                                        b_covers = set(neighbors[b].tolist())
                                        if all(u in a_covers or u in b_covers for u in U):
                                            if i1 > i2:
                                                i1, i2 = i2, i1
                                            selected[i1] = a
                                            selected[i2] = b
                                            cover_count[neighbors[a]] += 1
                                            cover_count[neighbors[b]] += 1
                                            ex_changed = True
                                            found = True
                                            break
                                elif len(remaining) <= 2:
                                    # b must cover all remaining
                                    for b in scored:
                                        if b == a:
                                            continue
                                        b_covers = set(neighbors[b].tolist())
                                        if all(u in a_covers or u in b_covers for u in U):
                                            if i1 > i2:
                                                i1, i2 = i2, i1
                                            selected[i1] = a
                                            selected[i2] = b
                                            cover_count[neighbors[a]] += 1
                                            cover_count[neighbors[b]] += 1
                                            ex_changed = True
                                            found = True
                                            break
                                    if found:
                                        break
                        
                        if found:
                            break
                
                # Restore
                cover_count[neighbors[cw1]] += 1
                cover_count[neighbors[cw2]] += 1
        
        # Update best
        cur_size = len(selected)
        if cur_size < best_size:
            best_size = cur_size
            best_set = selected.copy()
            
            # Polish
            if best_size <= 500 and time.time() < deadline - 15.0:
                polish_cover = np.zeros(total, dtype=np.int32)
                for cw in best_set:
                    polish_cover[neighbors[cw]] += 1
                
                polish_selected = best_set.copy()
                p_changed = True
                while p_changed and time.time() < deadline - 12.0:
                    p_changed = False
                    order = list(range(len(polish_selected)))
                    rng.shuffle(order)
                    for i in order:
                        if time.time() > deadline - 13.0:
                            break
                        cw = polish_selected[i]
                        if np.min(polish_cover[neighbors[cw]]) > 1:
                            polish_selected.pop(i)
                            polish_cover[neighbors[cw]] -= 1
                            p_changed = True
                            break
                
                if len(polish_selected) < best_size:
                    best_size = len(polish_selected)
                    best_set = polish_selected
        
        if best_size <= 486:
            break
    
    if best_set is None:
        return [list(range(n)) for _ in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
