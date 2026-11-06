# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    time_limit = 58.0  # safety margin before 60s kill

    n = 255
    # Build conflict bitsets
    conflict = [0] * n
    for i in range(n):
        ai = i + 2
        for j in range(i + 1, n):
            aj = j + 2
            l = ai * aj // math.gcd(ai, aj)
            if l <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    w = [1.0 / (i + 2) for i in range(n)]

    # Greedy initial solution to set a good lower bound
    greedy_set = set()
    greedy_mask = 0
    for i in range(n):
        if (conflict[i] & greedy_mask) == 0:
            greedy_set.add(i)
            greedy_mask |= (1 << i)
    greedy_value = sum(w[i] for i in greedy_set)

    best_value = greedy_value
    best_set = [i + 2 for i in sorted(greedy_set)]

    # For branch and bound, we need an efficient upper bound.
    # Key optimization: precompute for each position, the set of vertices from pos onward
    # that are "available" (don't conflict with each other is hard, but we can use a simple bound)
    
    # Actually, let's use a different approach: since n=255 and the graph is sparse-ish,
    # let's try a smarter branch and bound with vertex ordering by degree.
    
    # Alternative: use the fact that if we include a small number, it blocks many others.
    # Let's do a recursive search with aggressive pruning.
    
    # Precompute: for each vertex i, the list of vertices j > i that conflict with i
    # This helps in the "take" branch: we can skip all conflicting vertices.
    
    # Let's use an iterative deepening or a different strategy.
    # Given the time constraint, let's do a focused local search with 2-opt and 3-opt moves.
    
    current = set(greedy_set)
    current_value = greedy_value
    
    # Local search with multiple move types
    improved = True
    max_iterations = 500
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iter_count += 1
        
        # Try adding one vertex
        for i in range(n):
            if i in current:
                continue
            if (conflict[i] & greedy_mask) == 0:  # check against current mask
                # Actually need to check against current, not greedy_mask
                pass
            # Check if i can be added to current
            can_add = True
            for j in current:
                if (conflict[i] >> j) & 1:
                    can_add = False
                    break
            if can_add:
                current.add(i)
                current_value += w[i]
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove one, add one
        found_swap = False
        for rm in list(current):
            remaining = current - {rm}
            for add in range(n):
                if add in current:
                    continue
                can_add = True
                for j in remaining:
                    if (conflict[add] >> j) & 1:
                        can_add = False
                        break
                if can_add:
                    delta = w[add] - w[rm]
                    if delta > 1e-15:
                        current.discard(rm)
                        current.add(add)
                        current_value += delta
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                break
        
        if improved:
            continue
        
        # Try 2-swap: remove two, add two
        found_2swap = False
        current_list = list(current)
        for i1 in range(len(current_list)):
            for i2 in range(i1 + 1, len(current_list)):
                rm1, rm2 = current_list[i1], current_list[i2]
                remaining = current - {rm1, rm2}
                for add1 in range(n):
                    if add1 in current:
                        continue
                    can_add1 = True
                    for j in remaining:
                        if (conflict[add1] >> j) & 1:
                            can_add1 = False
                            break
                    if not can_add1:
                        continue
                    for add2 in range(add1 + 1, n):
                        if add2 in current or add2 == add1:
                            continue
                        can_add2 = True
                        for j in remaining:
                            if (conflict[add2] >> j) & 1:
                                can_add2 = False
                                break
                        if not can_add2:
                            continue
                        # Check add1 and add2 don't conflict
                        if (conflict[add1] >> add2) & 1:
                            continue
                        delta = w[add1] + w[add2] - w[rm1] - w[rm2]
                        if delta > 1e-15:
                            current.discard(rm1)
                            current.discard(rm2)
                            current.add(add1)
                            current.add(add2)
                            current_value += delta
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            if found_2swap:
                break
        
        if improved:
            continue
        
        # Try 1-remove-2-add
        found_r1a2 = False
        for rm in list(current):
            remaining = current - {rm}
            for add1 in range(n):
                if add1 in current:
                    continue
                can_add1 = True
                for j in remaining:
                    if (conflict[add1] >> j) & 1:
                        can_add1 = False
                        break
                if not can_add1:
                    continue
                for add2 in range(add1 + 1, n):
                    if add2 in current or add2 == add1:
                        continue
                    can_add2 = True
                    for j in remaining:
                        if (conflict[add2] >> j) & 1:
                            can_add2 = False
                            break
                    if not can_add2:
                        continue
                    if (conflict[add1] >> add2) & 1:
                        continue
                    delta = w[add1] + w[add2] - w[rm]
                    if delta > 1e-15:
                        current.discard(rm)
                        current.add(add1)
                        current.add(add2)
                        current_value += delta
                        improved = True
                        found_r1a2 = True
                        break
                if found_r1a2:
                    break
            if found_r1a2:
                break
        
        if improved:
            continue

    result = sorted(current)
    if not result:
        return [256]
    
    # Validate
    for idx, a in enumerate(result):
        for b in result[idx + 1:]:
            if math.lcm(a, b) <= 256:
                return [256]
    return result

# EVOLVE-BLOCK-END
