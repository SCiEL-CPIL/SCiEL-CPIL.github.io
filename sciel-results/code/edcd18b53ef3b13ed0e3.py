import random
import time
import itertools
import math

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n > 4, use Simulated Annealing to optimize the permutation order
    import itertools
    
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # Get standard order of permutations to start
    def get_standard_order(n):
        word = [1]
        for m in range(2, n + 1):
            p_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    p_list.append(window)
            new = []
            for p in p_list:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        
        order = []
        seen = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if w not in seen:
                seen.add(w)
                order.append(w)
        return order

    current_order = get_standard_order(n)
    
    # Precompute overlap costs? 
    # Overlap(p1, p2) depends on the specific pair. 
    # We can cache overlaps in a dict or numpy array if we map perms to indices.
    
    # Map permutations to indices for fast lookup and caching
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    # Re-index current_order to indices
    curr_indices = [perm_to_idx[p] for p in current_order]
    
    # Cache overlaps
    # N = 5040. N^2 is 25M. Too big for a full matrix in memory if using floats? 25M * 8 bytes = 200MB. Might be okay, but let's use a dict or sparse representation if needed.
    # Actually, we can compute overlap on the fly. It's O(n) which is 7. Very fast.
    
    def get_overlap_idx(i1, i2):
        p1 = perms[i1]
        p2 = perms[i2]
        max_k = n
        for k in range(max_k, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    def calc_total_cost(indices):
        total = 0
        for i in range(len(indices) - 1):
            total += n - get_overlap_idx(indices[i], indices[i+1])
        return total

    current_cost = calc_total_cost(curr_indices)
    
    start_time = time.time()
    time_limit = 110.0
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    best_indices = list(curr_indices)
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Generate a neighbor
        # Strategy: Random swap or 2-opt (reverse segment)
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(len(curr_indices)), 2)
            curr_indices[i], curr_indices[j] = curr_indices[j], curr_indices[i]
            
            # Calculate delta cost
            # Only edges (i-1, i), (i, i+1), (j-1, j), (j, j+1) change.
            # Note: if i and j are close, edges might overlap.
            
            def edge_cost(a, b):
                if a is None or b is None:
                    return 0
                return n - get_overlap_idx(a, b)
            
            # Old costs
            old_c1 = edge_cost(curr_indices[i-1] if i > 0 else None, curr_indices[i])
            old_c2 = edge_cost(curr_indices[i], curr_indices[i+1] if i < len(curr_indices)-1 else None)
            old_c3 = edge_cost(curr_indices[j-1] if j > 0 else None, curr_indices[j])
            old_c4 = edge_cost(curr_indices[j], curr_indices[j+1] if j < len(curr_indices)-1 else None)
            
            # Swap back to calculate new costs correctly? 
            # No, we have already swapped in curr_indices.
            # So current state is SWAPPED.
            # We need to compare SWAPPED cost vs UNSWAPPED cost.
            
            # Let's revert to calculate old costs properly, or just calculate new costs from current (swapped) state and compare to stored current_cost?
            # Easier: Calculate cost of the affected edges in the CURRENT (swapped) state.
            
            new_c1 = edge_cost(curr_indices[i-1] if i > 0 else None, curr_indices[i])
            new_c2 = edge_cost(curr_indices[i], curr_indices[i+1] if i < len(curr_indices)-1 else None)
            new_c3 = edge_cost(curr_indices[j-1] if j > 0 else None, curr_indices[j])
            new_c4 = edge_cost(curr_indices[j], curr_indices[j+1] if j < len(curr_indices)-1 else None)
            
            # Wait, the indices i and j refer to the positions. The values at those positions have swapped.
            # The edges are defined by positions.
            # Edge (i-1, i) involves value at i-1 and value at i.
            # In old state: val(i-1), val(i).
            # In new state: val(i-1), val(j) [since i and j swapped values].
            
            # This logic is tricky. Let's just do a full recalculation for simplicity if N is small enough?
            # N=5040. Full recalc is O(N) = 5040 overlap checks. Each check is O(7). ~35k ops.
            # 110 seconds / (35k ops) is plenty of iterations if we don't do full recalc every time.
            # But doing full recalc for every move is 5040 * 7 = 35k operations.
            # In Python, 35k ops is ~0.001s.
            # So we can do ~1000 iterations per second.
            # 110s * 1000 = 110,000 iterations. This is decent.
            
            # Let's stick to full recalculation for correctness and simplicity, but optimize the overlap function.
            
            # Actually, let's just use the full recalc function `calc_total_cost` for the neighbor.
            # It's simpler and less prone to edge case bugs in delta calculation.
            
            new_cost = calc_total_cost(curr_indices)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_indices = list(curr_indices)
            else:
                # Revert
                curr_indices[i], curr_indices[j] = curr_indices[j], curr_indices[i]
                
        else:
            # 2-opt: Reverse segment [i, j]
            i, j = sorted(random.sample(range(len(curr_indices)), 2))
            curr_indices[i:j+1] = reversed(curr_indices[i:j+1])
            
            new_cost = calc_total_cost(curr_indices)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_indices = list(curr_indices)
            else:
                # Revert
                curr_indices[i:j+1] = reversed(curr_indices[i:j+1])
                
        T *= cooling_rate
        # If stuck for a while, restart temperature?
        # Simple cooling is usually fine.

    # Reconstruct word from best_indices
    best_order = [perms[idx] for idx in best_indices]
    word = list(best_order[0])
    for i in range(1, len(best_order)):
        p_prev = best_order[i-1]
        p_curr = best_order[i]
        ov = get_overlap_idx(best_indices[i-1], best_indices[i])
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
