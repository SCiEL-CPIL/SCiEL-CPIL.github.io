# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 255  # numbers 2..256
    nums = list(range(2, 257))
    
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitset of vertices that conflict with i
    conflict_mask = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.lcm(nums[i], nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute reciprocal values
    recip = [1.0 / nums[i] for i in range(N)]
    
    # Pre-sort vertices by reciprocal descending
    sorted_by_recip = sorted(range(N), key=lambda x: -recip[x])
    
    # Greedy initialization
    def greedy(order):
        selected_mask = 0
        for v in order:
            if (selected_mask & conflict_mask[v]) == 0:
                selected_mask |= (1 << v)
        return selected_mask

    # Local Search with 1-for-1 and 2-for-1 exchanges
    def local_search(selected_mask, time_limit=5.0):
        sel_mask = selected_mask
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]
        
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_limit:
            improved = False
            
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
            
            # 1-for-1 Exchange: Remove v, add u
            for v in sel_list:
                rest_mask = sel_mask ^ (1 << v)
                for u in sorted_by_recip:
                    if (sel_mask >> u) & 1:
                        continue
                    if (conflict_mask[u] & rest_mask) == 0:
                        delta = recip[u] - recip[v]
                        if delta > 1e-12:
                            sel_mask = rest_mask | (1 << u)
                            best_val += delta
                            improved = True
                            break
                if improved:
                    break
            
            # 2-for-1 Exchange: Remove v1, v2, add u
            if not improved:
                sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
                for i_idx in range(len(sel_list)):
                    if time.time() - t_start > time_limit:
                        break
                    v1 = sel_list[i_idx]
                    for j_idx in range(i_idx + 1, len(sel_list)):
                        v2 = sel_list[j_idx]
                        base_mask = sel_mask ^ (1 << v1) ^ (1 << v2)
                        found = False
                        for u in sorted_by_recip:
                            if (sel_mask >> u) & 1:
                                continue
                            if (conflict_mask[u] & base_mask) != 0:
                                continue
                            if recip[u] > recip[v1] + recip[v2] + 1e-12:
                                sel_mask = base_mask | (1 << u)
                                best_val += recip[u] - recip[v1] - recip[v2]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
        
        return sel_mask, best_val

    # Exhaustive Polish
    def exhaustive_polish(selected_mask):
        sel_mask = selected_mask
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]
        
        # Try adding any valid u
        for u in range(N):
            if (sel_mask >> u) & 1:
                continue
            if (conflict_mask[u] & sel_mask) == 0:
                sel_mask |= (1 << u)
                best_val += recip[u]
        
        # Try swapping any v with any u (multiple passes)
        for _ in range(3):
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
            any_swap = False
            for v in sel_list:
                rest_mask = sel_mask ^ (1 << v)
                for u in sorted_by_recip:
                    if (sel_mask >> u) & 1:
                        continue
                    if (conflict_mask[u] & rest_mask) == 0:
                        if recip[u] > recip[v] + 1e-12:
                            sel_mask = rest_mask | (1 << u)
                            best_val += recip[u] - recip[v]
                            any_swap = True
                            break
            if not any_swap:
                break
        
        return sel_mask, best_val

    # Branch and Bound with tight upper bound
    def branch_and_bound(time_limit=15.0):
        """B&B on top vertices by reciprocal value."""
        best_sel = 0
        best_val = 0.0
        t_start = time.time()
        
        # Take top 50 vertices by reciprocal value
        top_verts = sorted_by_recip[:50]
        # Map vertex index to position in top_verts for fast lookup
        # Actually, let's just work with the list directly
        
        def _bb(idx, cur_sum, cur_mask):
            nonlocal best_sel, best_val
            if time.time() - t_start > time_limit:
                return
            if idx >= len(top_verts):
                if cur_sum > best_val:
                    best_val = cur_sum
                    best_sel = cur_mask
                return
            
            # Upper bound: cur_sum + sum of reciprocals of remaining vertices
            # that don't conflict with cur_mask
            ub = cur_sum
            for j in range(idx, len(top_verts)):
                v = top_verts[j]
                if (cur_mask >> v) & 1:
                    continue
                if (conflict_mask[v] & cur_mask) != 0:
                    continue
                ub += recip[v]
            
            if ub <= best_val + 1e-12:
                return
            
            v = top_verts[idx]
            
            # Try including v
            if (conflict_mask[v] & cur_mask) == 0:
                _bb(idx + 1, cur_sum + recip[v], cur_mask | (1 << v))
            
            # Try excluding v
            _bb(idx + 1, cur_sum, cur_mask)
        
        _bb(0, 0.0, 0)
        
        return best_sel, best_val

    # Multi-start
    best_result_mask = 0
    best_score = 0.0

    # Start 1: Natural order
    sel1 = greedy(list(range(N)))
    sel1, val1 = local_search(sel1, time_limit=8.0)
    if val1 > best_score:
        best_score = val1
        best_result_mask = sel1

    # Start 2: Reverse order
    sel2 = greedy(list(range(N - 1, -1, -1)))
    sel2, val2 = local_search(sel2, time_limit=8.0)
    if val2 > best_score:
        best_score = val2
        best_result_mask = sel2

    # Start 3: Random orderings
    rng = random.Random(42)
    for _ in range(8):
        if time.time() - t0 > DEADLINE - 25:
            break
        order = list(range(N))
        rng.shuffle(order)
        sel = greedy(order)
        sel, val = local_search(sel, time_limit=4.0)
        if val > best_score:
            best_score = val
            best_result_mask = sel

    # Start 4: Branch and Bound
    if time.time() - t0 < DEADLINE - 20:
        bb_mask, bb_val = branch_and_bound(time_limit=15.0)
        if bb_val > best_score:
            best_score = bb_val
            best_result_mask = bb_mask
        # Polish the B&B result
        if best_result_mask != 0:
            best_result_mask, best_score = local_search(best_result_mask, time_limit=5.0)

    # Final Polish
    if time.time() - t0 < DEADLINE - 5:
        best_result_mask, best_score = local_search(best_result_mask, time_limit=10.0)
        best_result_mask, best_score = exhaustive_polish(best_result_mask)

    # Convert to list of actual numbers
    result = [nums[i] for i in range(N) if (best_result_mask >> i) & 1]
    return sorted(result) if result else [2]
# EVOLVE-BLOCK-END
