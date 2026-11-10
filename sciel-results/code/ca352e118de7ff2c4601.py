import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute 3-APs
    aps = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c <= b or c > N:
                continue
            aps.add((a, b, c))
    
    # conflict_pairs[x] = list of (y, z) such that {x, y, z} is a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    for a, b, c in aps:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # affects[x] = list of (other_x, other_y) such that (other_x, other_y) is in conflict_pairs[x]
    affects = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            affects[y].append((x, z))
            affects[z].append((x, y))
    
    def make_state(S_mask):
        in_S = [False] * (N + 1)
        block_count = [0] * (N + 1)
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                in_S[x] = True
        for x in range(1, N + 1):
            for y, z in conflict_pairs[x]:
                if in_S[y] and in_S[z]:
                    block_count[x] += 1
        return in_S, block_count
    
    def add_to_state(x, in_S, block_count):
        in_S[x] = True
        for other_x, other_y in affects[x]:
            if in_S[other_y]:
                block_count[other_x] += 1
    
    def remove_from_state(x, in_S, block_count):
        for other_x, other_y in affects[x]:
            if in_S[other_y]:
                block_count[other_x] -= 1
        in_S[x] = False
    
    def can_add(x, in_S, block_count):
        return not in_S[x] and block_count[x] == 0
    
    def get_sum(in_S):
        total = 0.0
        for x in range(1, N + 1):
            if in_S[x]:
                total += recip[x]
        return total
    
    def greedy_fill(forbidden, in_S, block_count):
        for x in range(1, N + 1):
            if forbidden & (1 << x):
                continue
            if can_add(x, in_S, block_count):
                add_to_state(x, in_S, block_count)
    
    def greedy_fill_descending(forbidden, in_S, block_count):
        for x in range(N, 0, -1):
            if forbidden & (1 << x):
                continue
            if can_add(x, in_S, block_count):
                add_to_state(x, in_S, block_count)
    
    def local_search(in_S, block_count, sum_val, max_time):
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Phase 1: Try to add any element
            for x in range(1, N + 1):
                if can_add(x, in_S, block_count):
                    add_to_state(x, in_S, block_count)
                    sum_val += recip[x]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: 1-swap
            best_gain = 0
            best_x = -1
            best_y = -1
            
            for y in range(1, N + 1):
                if in_S[y]:
                    continue
                for x in range(N, 0, -1):
                    if not in_S[x]:
                        continue
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        break
                    remove_from_state(x, in_S, block_count)
                    valid = block_count[y] == 0
                    add_to_state(x, in_S, block_count)
                    if valid:
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
                        break
            
            if best_y != -1:
                remove_from_state(best_x, in_S, block_count)
                add_to_state(best_y, in_S, block_count)
                sum_val += best_gain
                improved = True
                continue
            
            # Phase 3: 1-remove + re-greedy
            best_remove_gain = -1
            best_remove_x = -1
            best_remove_in_S = None
            best_remove_block = None
            
            for x in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if not in_S[x]:
                    continue
                
                in_S_copy = in_S[:]
                block_copy = block_count[:]
                
                remove_from_state(x, in_S, block_count)
                greedy_fill(0, in_S, block_count)
                
                new_sum = sum_val - recip[x]
                for i in range(1, N + 1):
                    if in_S[i] and not in_S_copy[i]:
                        new_sum += recip[i]
                
                gain = new_sum - sum_val
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_x = x
                    best_remove_in_S = in_S[:]
                    best_remove_block = block_count[:]
                
                in_S[:] = in_S_copy
                block_count[:] = block_copy
            
            if best_remove_x != -1:
                in_S[:] = best_remove_in_S
                block_count[:] = best_remove_block
                sum_val += best_remove_gain
                improved = True
                continue
            
            # Phase 4: 2-swap
            if (time.time() - start_time) < max_time - 2:
                S_list = [x for x in range(1, N + 1) if in_S[x]]
                not_S_list = [x for x in range(1, N + 1) if not in_S[x]]
                
                if len(S_list) >= 2 and len(not_S_list) >= 2:
                    num_tries = min(50, max(10, int((max_time - (time.time() - start_time)) * 3)))
                    for _ in range(num_tries):
                        if (time.time() - start_time) > max_time:
                            break
                        x1, x2 = random.sample(S_list, 2)
                        y1, y2 = random.sample(not_S_list, 2)
                        
                        gain = recip[y1] + recip[y2] - recip[x1] - recip[x2]
                        if gain <= 1e-15:
                            continue
                        
                        in_S_copy = in_S[:]
                        block_copy = block_count[:]
                        
                        remove_from_state(x1, in_S, block_count)
                        remove_from_state(x2, in_S, block_count)
                        
                        valid = block_count[y1] == 0
                        if valid:
                            add_to_state(y1, in_S, block_count)
                            valid = block_count[y2] == 0
                            if valid:
                                add_to_state(y2, in_S, block_count)
                        
                        if valid:
                            sum_val += gain
                            improved = True
                            break
                        
                        in_S[:] = in_S_copy
                        block_count[:] = block_copy
            
            if not improved:
                # Random perturbation
                if (time.time() - start_time) > max_time:
                    break
                S_list = [x for x in range(1, N + 1) if in_S[x]]
                if not S_list:
                    break
                x = random.choice(S_list)
                in_S_copy = in_S[:]
                block_copy = block_count[:]
                
                remove_from_state(x, in_S, block_count)
                greedy_fill(0, in_S, block_count)
                new_sum = sum_val - recip[x]
                for i in range(1, N + 1):
                    if in_S[i] and not in_S_copy[i]:
                        new_sum += recip[i]
                
                if new_sum > sum_val:
                    sum_val = new_sum
                    improved = True
                else:
                    in_S[:] = in_S_copy
                    block_count[:] = block_copy
        
        return in_S, block_count, sum_val
    
    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    in_S, block = make_state(0)
    greedy_fill(0, in_S, block)
    S_asc_mask = 0
    for x in range(1, N + 1):
        if in_S[x]:
            S_asc_mask |= (1 << x)
    initializations.append(S_asc_mask)
    
    # Greedy descending
    in_S, block = make_state(0)
    greedy_fill_descending(0, in_S, block)
    S_desc_mask = 0
    for x in range(1, N + 1):
        if in_S[x]:
            S_desc_mask |= (1 << x)
    initializations.append(S_desc_mask)
    
    # Randomized greedy
    for trial in range(15):
        if (time.time() - start_time) > deadline - 25:
            break
        random.seed(trial * 12345 + 67890)
        in_S, block = make_state(0)
        for x in range(1, N + 1):
            if random.random() < 0.08 and x > 3:
                continue
            if can_add(x, in_S, block):
                add_to_state(x, in_S, block)
        mask = 0
        for x in range(1, N + 1):
            if in_S[x]:
                mask |= (1 << x)
        initializations.append(mask)
    
    # Step 2: Local Search on each initialization
    best_S_mask = 0
    best_sum = 0.0
    best_in_S = None
    best_block = None
    
    remaining_init_time = deadline - 10 - (time.time() - start_time)
    if remaining_init_time < 5:
        remaining_init_time = 5
    
    time_per_ls = max(3.0, remaining_init_time / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 3:
            break
        
        in_S, block = make_state(S_init)
        sum_val = get_sum(in_S)
        
        S_time = min(time_per_ls, remaining_time)
        in_S, block, sum_val = local_search(in_S, block, sum_val, start_time + S_time)
        
        if sum_val > best_sum:
            best_sum = sum_val
            best_in_S = in_S
            best_block = block
            best_S_mask = 0
            for x in range(1, N + 1):
                if in_S[x]:
                    best_S_mask |= (1 << x)
    
    # Step 3: Final Polish
    if best_in_S is not None:
        remaining = deadline - 3 - (time.time() - start_time)
        if remaining > 5:
            in_S, block, sum_final = local_search(best_in_S, best_block, best_sum, start_time + remaining)
            if sum_final > best_sum:
                best_sum = sum_final
                best_in_S = in_S
                best_block = block
    
    # Convert to list
    if best_in_S is not None:
        result = [x for x in range(1, N + 1) if best_in_S[x]]
    else:
        result = [x for x in range(1, N + 1) if best_S_mask & (1 << x)]
    
    # Validate and repair if needed
    S_set = set(result)
    changed = True
    while changed:
        changed = False
        for a in sorted(S_set):
            for b in sorted(S_set):
                if a >= b:
                    continue
                s = a + b
                if s % 2 == 0:
                    mid = s // 2
                    if mid in S_set and mid != a and mid != b:
                        vals = [a, b, mid]
                        worst = max(vals)
                        S_set.discard(worst)
                        changed = True
                        break
            if changed:
                break
    
    result = sorted(S_set)
    
    if not result:
        in_S, block = make_state(0)
        greedy_fill(0, in_S, block)
        result = [x for x in range(1, N + 1) if in_S[x]]
    
    return result
