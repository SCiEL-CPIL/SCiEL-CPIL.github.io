"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    N = 256
    
    # Precompute for each x, the list of pairs (a,c) such that {a,x,c} is a 3-AP
    # x as middle: (x-d, x+d)
    # x as left end: (x+d, x+2d)
    # x as right end: (x-2d, x-d)
    # For can_add(mask, x): check if any pair in conflicts[x] is both in mask
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as middle
        for d in range(1, x):
            a = x - d
            c = x + d
            if c > N:
                break
            conflicts[x].append((a, c))
        # x as left end
        for d in range(1, (N - x + 1) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if c > N:
                break
            conflicts[x].append((b, c))
        # x as right end
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            conflicts[x].append((a, b))
    
    def can_add(mask, x):
        """Check if adding x to mask creates a 3-term AP using precomputed conflicts."""
        for a, c in conflicts[x]:
            if (mask & (1 << a)) and (mask & (1 << c)):
                return False
        return True
    
    def reciprocal_sum_from_mask(m):
        s = 0.0
        for x in range(1, N + 1):
            if m & (1 << x):
                s += 1.0 / x
        return s
    
    def local_search(mask, best_score, max_time):
        """Hill climbing with add, swap, and 2-opt moves."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            sel_list = [x for x in range(1, N + 1) if mask & (1 << x)]
            unsel_list = [x for x in range(1, N + 1) if not (mask & (1 << x))]
            
            # Try add moves
            for x in unsel_list:
                if time.time() > max_time:
                    return mask, best_score
                if can_add(mask, x):
                    new_mask = mask | (1 << x)
                    new_score = reciprocal_sum_from_mask(new_mask)
                    if new_score > best_score + 1e-15:
                        mask = new_mask
                        best_score = new_score
                        improved = True
                        sel_list.append(x)
                        unsel_list.remove(x)
            
            if time.time() > max_time:
                return mask, best_score
            
            # Try swap moves (remove one selected, add one unselected with smaller value)
            for a in sel_list:
                if time.time() > max_time:
                    return mask, best_score
                for b in unsel_list:
                    if b >= a:
                        continue
                    new_mask = mask & ~(1 << a)
                    if can_add(new_mask, b):
                        new_mask |= (1 << b)
                        new_score = reciprocal_sum_from_mask(new_mask)
                        if new_score > best_score + 1e-15:
                            mask = new_mask
                            best_score = new_score
                            improved = True
                            sel_list.remove(a)
                            sel_list.append(b)
                            unsel_list.remove(b)
                            unsel_list.append(a)
                            break
            
            if time.time() > max_time:
                return mask, best_score
            
            # Try 2-opt moves: remove two selected, add two unselected
            # Only try if potential gain: 1/b1 + 1/b2 > 1/a1 + 1/a2
            if len(sel_list) >= 2 and len(unsel_list) >= 2:
                # Sort selected descending (larger values = smaller reciprocals, good candidates to remove)
                sel_desc = sorted(sel_list, reverse=True)
                # Sort unselected ascending (smaller values = larger reciprocals, good candidates to add)
                unsel_asc = sorted(unsel_list)
                
                # Try combinations of 2 to remove and 2 to add
                # Limit to top candidates for efficiency
                n_remove = min(len(sel_desc), 10)
                n_add = min(len(unsel_asc), 10)
                
                for i in range(n_remove):
                    if time.time() > max_time:
                        return mask, best_score
                    a1 = sel_desc[i]
                    for j in range(i + 1, n_remove):
                        a2 = sel_desc[j]
                        # Potential max gain: remove a1,a2, add best two unselected
                        max_gain = 1.0 / unsel_asc[0] + 1.0 / (unsel_asc[1] if len(unsel_asc) > 1 else N) - 1.0 / a1 - 1.0 / a2
                        if max_gain <= 1e-15:
                            continue
                        for k in range(n_add):
                            if time.time() > max_time:
                                return mask, best_score
                            b1 = unsel_asc[k]
                            for l in range(k + 1, n_add):
                                b2 = unsel_asc[l]
                                if 1.0 / b1 + 1.0 / b2 <= 1.0 / a1 + 1.0 / a2 + 1e-15:
                                    continue
                                new_mask = mask & ~(1 << a1) & ~(1 << a2)
                                if can_add(new_mask, b1) and can_add(new_mask | (1 << b1), b2):
                                    new_mask |= (1 << b1) | (1 << b2)
                                    new_score = reciprocal_sum_from_mask(new_mask)
                                    if new_score > best_score + 1e-15:
                                        mask = new_mask
                                        best_score = new_score
                                        improved = True
                                        # Update lists
                                        sel_list.remove(a1)
                                        sel_list.remove(a2)
                                        sel_list.extend([b1, b2])
                                        unsel_list.remove(b1)
                                        unsel_list.remove(b2)
                                        unsel_list.extend([a1, a2])
                                        return mask, best_score  # Restart local search
        
        return mask, best_score
    
    # Greedy: add numbers from 1 to 256 if they don't create a 3-AP
    mask = 0
    for x in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        if can_add(mask, x):
            mask |= (1 << x)
    
    best_mask = mask
    best_score = reciprocal_sum_from_mask(mask)
    
    # Local search on initial solution
    best_mask, best_score = local_search(best_mask, best_score, deadline - 15)
    
    # Try multiple random restarts with different greedy orderings if time permits
    if time.time() < deadline - 20:
        rng = random.Random(42)
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Weighted random greedy: prioritize smaller numbers (higher reciprocal value)
            order = list(range(1, N + 1))
            # Bias towards smaller numbers
            weights = [1.0 / x for x in order]
            # Shuffle with bias
            for i in range(len(order) - 1, 0, -1):
                # Choose from first i+1 elements with probability proportional to weight
                total_w = sum(weights[j] for j in range(i + 1))
                r = rng.random() * total_w
                cum = 0
                for j in range(i + 1):
                    cum += weights[j]
                    if r <= cum:
                        order[i], order[j] = order[j], order[i]
                        weights[i], weights[j] = weights[j], weights[i]
                        break
            
            rmask = 0
            for x in order:
                if time.time() > deadline - 10:
                    break
                if can_add(rmask, x):
                    rmask |= (1 << x)
            
            rscore = reciprocal_sum_from_mask(rmask)
            if rscore > best_score:
                best_mask = rmask
                best_score = rscore
            
            # Local search on this restart
            best_mask, best_score = local_search(best_mask, best_score, deadline - 10)
    
    # Final result
    result = [x for x in range(1, N + 1) if best_mask & (1 << x)]
    return sorted(result)
# EVOLVE-BLOCK-END
