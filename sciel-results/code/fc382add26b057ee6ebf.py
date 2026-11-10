import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp_bits[a] has bit b set iff a and b are comparable (a != b, one is proper subset of other)
    comp_bits = [0] * num_masks
    
    for a in range(num_masks):
        bits = 0
        if a != 0:
            # Proper subsets of a
            s = (a - 1) & a
            while True:
                bits |= (1 << s)
                if s == 0:
                    break
                s = (s - 1) & a
        # Proper supersets of a
        free = (~a) & full_mask
        if free != 0:
            s = (free - 1) & free
            while True:
                sup = a | s
                bits |= (1 << sup)
                if s == 0:
                    break
                s = (s - 1) & free
        comp_bits[a] = bits
    
    def greedy_init():
        # Start with a chain: 0, 1, 3, 7, ..., full_mask
        chain = [0]
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            chain.append(cur)
        selected = list(chain)
        selected_set = set(selected)
        selected_bits = 0
        for s in selected:
            selected_bits |= (1 << s)
        
        # Greedily add m-13 more sets
        for _ in range(m - 13):
            best_gain = -1
            best_mask = -1
            for candidate in range(num_masks):
                if candidate in selected_set:
                    continue
                gain = (comp_bits[candidate] & selected_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_mask = candidate
            if best_mask == -1:
                break
            selected.append(best_mask)
            selected_set.add(best_mask)
            selected_bits |= (1 << best_mask)
        return selected
    
    def local_search(initial_masks):
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        # Calculate initial score
        cur_score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (comp_bits[current[i]] >> current[j]) & 1:
                    cur_score += 1
        
        best_score = cur_score
        best_solution = list(current)
        
        stuck = 0
        max_steps = 200000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            idx = random.randint(0, m - 1)
            old = current[idx]
            
            # Propose new mask
            op = random.random()
            new = old
            if op < 0.35:
                # Add a bit
                zeros = [i for i in range(n) if not (old & (1 << i))]
                if zeros:
                    bit = random.choice(zeros)
                    new = old | (1 << bit)
            elif op < 0.70:
                # Remove a bit
                ones = [i for i in range(n) if (old & (1 << i))]
                if ones:
                    bit = random.choice(ones)
                    new = old ^ (1 << bit)
            elif op < 0.85:
                # Swap a bit
                ones = [i for i in range(n) if (old & (1 << i))]
                zeros = [i for i in range(n) if not (old & (1 << i))]
                if ones and zeros:
                    bo = random.choice(ones)
                    bz = random.choice(zeros)
                    new = (old ^ (1 << bo)) | (1 << bz)
            else:
                # Random jump
                new = random.randint(0, full_mask)
            
            if new == old or new in current_set:
                continue
            
            # Fast delta calculation using precomputed bitsets
            others = selected_bits & ~(1 << old)
            delta = (comp_bits[new] & others).bit_count() - (comp_bits[old] & others).bit_count()
            
            if delta > 0:
                current_set.remove(old)
                current_set.add(new)
                selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                current[idx] = new
                cur_score += delta
                stuck = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_solution = list(current)
            else:
                stuck += 1
                if stuck > 200:
                    # Random kick
                    idx = random.randint(0, m - 1)
                    old_kick = current[idx]
                    while True:
                        new_kick = random.randint(0, full_mask)
                        if new_kick not in current_set:
                            break
                    current_set.remove(old_kick)
                    current_set.add(new_kick)
                    selected_bits = (selected_bits & ~(1 << old_kick)) | (1 << new_kick)
                    current[idx] = new_kick
                    # Recalculate score
                    cur_score = 0
                    for i in range(m):
                        for j in range(i + 1, m):
                            if (comp_bits[current[i]] >> current[j]) & 1:
                                cur_score += 1
                    stuck = 0
        
        return best_score, best_solution
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    # Greedy chain initialization + local search
    greedy_solution = greedy_init()
    gs, gsol = local_search(greedy_solution)
    if gs > best_global_score:
        best_global_score = gs
        best_global_solution = gsol
    
    # Additional random restarts if time permits
    for r in range(3):
        if time.time() > deadline - 15.0:
            break
        random_solution = random.sample(range(num_masks), m)
        rs, rsol = local_search(random_solution)
        if rs > best_global_score:
            best_global_score = rs
            best_global_solution = rsol
    
    # Convert to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
