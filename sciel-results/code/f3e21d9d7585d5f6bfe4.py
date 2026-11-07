import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = P['n']
    max_val = n

    # Precompute pairs for sum cases
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))

    def is_valid_addition(x, mask):
        """
        Check if adding x to set represented by mask is valid.
        mask: bitset where bit i-1 is 1 if i is in S.
        """
        if mask & (1 << (x - 1)):
            return True

        # Case 1: x acts as a member 'a' in a triple (a,b,c)
        # U_x = { y in S | x+y in S }
        U_x = mask & (mask >> x)
        
        m = U_x
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            
            V_b = mask & (mask >> b_val)
            common = U_x & V_b
            if common & ~lsb_b:
                return False

        # Case 2: x acts as a sum a+b = x, with a,b in S.
        # Optimize: first check if there are at least 2 elements in S that sum to x
        # by checking pairs_sum[x] against mask
        pairs = pairs_sum[x]
        if not pairs:
            return True
        
        # Quick check: count how many pairs have both elements in mask
        # If fewer than 1 pair exists, no violation possible
        valid_pairs = []
        for a, b in pairs:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                valid_pairs.append((a, b))
        
        if len(valid_pairs) < 1:
            return True
        
        # For each valid pair, check if there's a c in S\{a,b} with a+c in S and b+c in S
        for a, b in valid_pairs:
            V_a = mask & (mask >> a)
            V_b = mask & (mask >> b)
            common = V_a & V_b
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False

        return True

    def popcount(x):
        return bin(x).count('1')

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    # ---- Structural start with better initial set ----
    # Try multiple structural starts and pick the best
    structural_starts = []
    
    # Start 1: High half + upper quarter of low half
    K = 64
    H = list(range(K, max_val + 1))
    max_L = (K - 1) // 2
    start_L = max_L // 2 + 1
    L1 = list(range(start_L, max_L + 1))
    structural_starts.append(H + L1)
    
    # Start 2: High half + sparser low
    L2 = list(range(16, 32, 2))  # even numbers 16-30
    structural_starts.append(H + L2)
    
    # Start 3: High half + different low range
    L3 = list(range(20, 32))
    structural_starts.append(H + L3)
    
    # Find best structural start
    best_mask = 0
    best_size = 0
    for s in structural_starts:
        mask = 0
        for v in s:
            mask |= (1 << (v - 1))
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    def local_search(initial_mask, time_limit):
        """Run local search from initial_mask, return best mask found."""
        current_mask = initial_mask
        current_size = popcount(current_mask)
        current_elements = elements_of(current_mask)
        unused_elements = set(unused_of(current_mask))
        
        local_best_mask = current_mask
        local_best_size = current_size
        
        no_progress = 0
        max_no_progress = 80
        
        while time.time() < time_limit:
            # 1. Greedy Add - try ALL unused elements
            if unused_elements:
                added = False
                for x in sorted(unused_elements):
                    if is_valid_addition(x, current_mask):
                        current_mask |= (1 << (x - 1))
                        current_size += 1
                        unused_elements.remove(x)
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_mask = current_mask
                        added = True
                        no_progress = 0
                        break
                if added:
                    continue
            
            # 2. 1-out, 2-in Swap
            if current_elements and len(unused_elements) >= 2:
                x_out = random.choice(current_elements)
                mask_trial = current_mask & ~(1 << (x_out - 1))
                
                valid_adds = []
                for x in unused_elements:
                    if is_valid_addition(x, mask_trial):
                        valid_adds.append(x)
                
                if len(valid_adds) >= 2:
                    random.shuffle(valid_adds)
                    found_swap = False
                    max_checks = 30
                    checks = 0
                    for i in range(len(valid_adds)):
                        if checks > max_checks:
                            break
                        x1 = valid_adds[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_adds)):
                            if checks > max_checks:
                                break
                            checks += 1
                            x2 = valid_adds[j]
                            if is_valid_addition(x2, mask_1):
                                current_mask = mask_1 | (1 << (x2 - 1))
                                current_size += 1
                                current_elements.remove(x_out)
                                current_elements.append(x1)
                                current_elements.append(x2)
                                unused_elements.remove(x1)
                                unused_elements.remove(x2)
                                unused_elements.add(x_out)
                                
                                if current_size > local_best_size:
                                    local_best_size = current_size
                                    local_best_mask = current_mask
                                
                                found_swap = True
                                no_progress = 0
                                break
                        if found_swap:
                            break
                    
                    if found_swap:
                        continue
            
            # 3. Perturbation - remove 1-2 random elements
            if current_elements:
                num_remove = 1 if random.random() < 0.7 else 2
                for _ in range(min(num_remove, len(current_elements))):
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
                    unused_elements.add(x_out)
                no_progress += 1
            
            if no_progress > max_no_progress:
                current_mask = local_best_mask
                current_size = local_best_size
                current_elements = elements_of(current_mask)
                unused_elements = set(unused_of(current_mask))
                no_progress = 0
        
        # Final polish
        if time.time() < time_limit:
            final_mask = local_best_mask
            final_unused = set(unused_of(final_mask))
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for x in sorted(final_unused):
                    if is_valid_addition(x, final_mask):
                        final_mask |= (1 << (x - 1))
                        final_unused.remove(x)
                        changed = True
            
            if popcount(final_mask) > local_best_size:
                local_best_mask = final_mask
                local_best_size = popcount(final_mask)
        
        return local_best_mask, local_best_size

    # Run multiple restarts with different random seeds
    best_mask = 0
    best_size = 0
    
    # First, run local search on the best structural start
    mask1, size1 = local_search(best_mask, deadline - 30.0)
    if size1 > best_size:
        best_size = size1
        best_mask = mask1
    
    # If we have time, try a few more restarts from perturbed versions
    restarts_done = 1
    while time.time() < deadline - 5.0 and restarts_done < 5:
        # Perturb the best solution slightly
        perturbed = best_mask
        elems = elements_of(perturbed)
        if elems:
            # Remove 2-3 random elements
            num_remove = random.randint(2, 3)
            for _ in range(min(num_remove, len(elems))):
                x = random.choice(elems)
                perturbed &= ~(1 << (x - 1))
        
        mask_r, size_r = local_search(perturbed, deadline - 5.0)
        if size_r > best_size:
            best_size = size_r
            best_mask = mask_r
        restarts_done += 1

    # Final polish on best
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_unused = set(unused_of(final_mask))
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for x in sorted(final_unused):
                if is_valid_addition(x, final_mask):
                    final_mask |= (1 << (x - 1))
                    final_unused.remove(x)
                    changed = True
        
        if popcount(final_mask) > best_size:
            best_mask = final_mask
            best_size = popcount(final_mask)

    # Convert best_mask to list of ints
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)
