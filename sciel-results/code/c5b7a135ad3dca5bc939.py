import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    n_max = 256

    # Precompute conflict masks: bit j set in mask[i] means i and j conflict
    conflict_mask = [0] * (n_max + 1)
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Precompute list of non-conflicting pairs for faster move generation
    # For each element, store list of elements it can coexist with
    compatible_with = [[] for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i + 1, n_max + 1):
            if not (conflict_mask[i] & (1 << j)):
                compatible_with[i].append(j)
                compatible_with[j].append(i)

    def popcount(x):
        return x.bit_count()

    def can_add(mask, x):
        return (mask & conflict_mask[x]) == 0

    # Greedy completion: add all possible elements iteratively until fixpoint
    def greedy_complete(mask):
        changed = True
        while changed:
            changed = False
            for i in range(1, n_max + 1):
                bit = 1 << i
                if not (mask & bit):
                    if (mask & conflict_mask[i]) == 0:
                        mask |= bit
                        changed = True
        return mask

    # Multi-start greedy to find good initial solution
    all_elements = list(range(1, n_max + 1))
    evens = list(range(2, n_max + 1, 2))
    odds = list(range(1, n_max + 1, 2))

    random.seed(42)
    best_mask = 0
    best_size = 0

    # Start from all odds (known good base)
    base_mask = 0
    for i in odds:
        base_mask |= (1 << i)
    best_mask = greedy_complete(base_mask)
    best_size = popcount(best_mask)

    # Try multiple random greedy orderings
    for _ in range(80):
        if time.time() > deadline - 15.0:
            break
        cur = 0
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        for e in shuffled:
            if can_add(cur, e):
                cur |= (1 << e)
        cur = greedy_complete(cur)
        sz = popcount(cur)
        if sz > best_size:
            best_size = sz
            best_mask = cur

    # Also try starting from random subsets of various sizes
    for trial_size in [48, 64, 96, 128]:
        for _ in range(15):
            if time.time() > deadline - 15.0:
                break
            cur = 0
            sample = random.sample(all_elements, min(trial_size, n_max))
            for e in sample:
                if can_add(cur, e):
                    cur |= (1 << e)
            cur = greedy_complete(cur)
            sz = popcount(cur)
            if sz > best_size:
                best_size = sz
                best_mask = cur

    # Simulated Annealing with optimized move generation
    current_mask = best_mask
    current_size = best_size

    # Precompute for each element: which elements can be added if this is removed
    # and which elements can be removed
    T = 8.0
    T_min = 0.001
    cooling_rate = 0.997
    steps = 0
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline - 10.0 and T > T_min and no_improve_count < max_no_improve:
        steps += 1
        r = random.random()
        new_mask = current_mask
        delta = 0
        valid_move = False

        if r < 0.30:
            # Add a random element
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if not (current_mask & bit_x):
                if (current_mask & conflict_mask[x]) == 0:
                    new_mask = current_mask | bit_x
                    delta = 1
                    valid_move = True
        elif r < 0.50:
            # Remove a random element
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if current_mask & bit_x:
                new_mask = current_mask & ~bit_x
                delta = -1
                valid_move = True
        elif r < 0.75:
            # Swap: remove x, add y
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if current_mask & bit_x:
                y = random.randint(1, n_max)
                bit_y = 1 << y
                if not (current_mask & bit_y):
                    mask_without = current_mask & ~bit_x
                    if (mask_without & conflict_mask[y]) == 0:
                        new_mask = mask_without | bit_y
                        delta = 0
                        valid_move = True
        else:
            # 2-swap: remove x1,x2 add y1,y2
            x1 = random.randint(1, n_max)
            x2 = random.randint(1, n_max)
            if x1 != x2:
                bit_x1 = 1 << x1
                bit_x2 = 1 << x2
                if (current_mask & bit_x1) and (current_mask & bit_x2):
                    y1 = random.randint(1, n_max)
                    y2 = random.randint(1, n_max)
                    if y1 != y2:
                        bit_y1 = 1 << y1
                        bit_y2 = 1 << y2
                        if not (current_mask & bit_y1) and not (current_mask & bit_y2):
                            mask_without = current_mask & ~bit_x1 & ~bit_x2
                            if (mask_without & conflict_mask[y1]) == 0 and (mask_without & conflict_mask[y2]) == 0:
                                # Check y1 and y2 don't conflict with each other
                                if (conflict_mask[y1] & bit_y2) == 0:
                                    new_mask = mask_without | bit_y1 | bit_y2
                                    delta = 0
                                    valid_move = True

        if not valid_move:
            continue

        new_size = new_mask.bit_count()

        if new_size > current_size:
            current_mask = new_mask
            current_size = new_size
            no_improve_count = 0
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        elif new_size < current_size:
            if random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_size = new_size
                no_improve_count += 1
        else:
            current_mask = new_mask
            no_improve_count += 1

        T *= cooling_rate

        # Periodic greedy completion
        if steps % 15 == 0:
            completed = greedy_complete(current_mask)
            csz = completed.bit_count()
            if csz > current_size:
                current_mask = completed
                current_size = csz
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final local search: systematic 1-swap and 2-swap improvement
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # 1-swap: try to replace x with y to improve
        for x in range(1, n_max + 1):
            if time.time() > deadline - 3.0:
                break
            bit_x = 1 << x
            if not (best_mask & bit_x):
                continue
            mask_without_x = best_mask & ~bit_x
            for y in range(1, n_max + 1):
                if time.time() > deadline - 3.0:
                    break
                bit_y = 1 << y
                if best_mask & bit_y:
                    continue
                if (mask_without_x & conflict_mask[y]) == 0:
                    candidate = mask_without_x | bit_y
                    cand_size = candidate.bit_count()
                    if cand_size > best_size:
                        best_mask = candidate
                        best_size = cand_size
                        improved = True
                        break
                    elif cand_size == best_size:
                        completed = greedy_complete(candidate)
                        csz = completed.bit_count()
                        if csz > best_size:
                            best_mask = completed
                            best_size = csz
                            improved = True
                            break
        
        # 2-swap: try to replace x1,x2 with y1,y2 to improve
        if not improved:
            for x1 in range(1, n_max + 1):
                if time.time() > deadline - 3.0:
                    break
                bit_x1 = 1 << x1
                if not (best_mask & bit_x1):
                    continue
                for x2 in range(x1 + 1, n_max + 1):
                    if time.time() > deadline - 3.0:
                        break
                    bit_x2 = 1 << x2
                    if not (best_mask & bit_x2):
                        continue
                    mask_without = best_mask & ~bit_x1 & ~bit_x2
                    for y1 in range(1, n_max + 1):
                        if time.time() > deadline - 3.0:
                            break
                        bit_y1 = 1 << y1
                        if best_mask & bit_y1:
                            continue
                        if (mask_without & conflict_mask[y1]) != 0:
                            continue
                        for y2 in range(y1 + 1, n_max + 1):
                            if time.time() > deadline - 3.0:
                                break
                            bit_y2 = 1 << y2
                            if best_mask & bit_y2:
                                continue
                            if (mask_without & conflict_mask[y2]) != 0:
                                continue
                            if (conflict_mask[y1] & bit_y2) != 0:
                                continue
                            candidate = mask_without | bit_y1 | bit_y2
                            cand_size = candidate.bit_count()
                            if cand_size > best_size:
                                best_mask = candidate
                                best_size = cand_size
                                improved = True
                                break
                            elif cand_size == best_size:
                                completed = greedy_complete(candidate)
                                csz = completed.bit_count()
                                if csz > best_size:
                                    best_mask = completed
                                    best_size = csz
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final greedy completion
    final_mask = greedy_complete(best_mask)

    # Convert to sorted list
    result = []
    for i in range(1, n_max + 1):
        if final_mask & (1 << i):
            result.append(i)

    return result
