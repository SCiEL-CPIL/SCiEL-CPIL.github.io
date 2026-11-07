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

    # Precompute for each x, the list of (a,b) pairs where a+b=x
    # and for each x, pairs (b,c) where b+c=x
    # Also precompute bit masks for quick checks
    
    def popcount(x):
        return bin(x).count('1')

    def is_valid_addition(x, mask):
        """
        Check if adding x to set represented by mask is valid.
        mask: bitset where bit i-1 is 1 if i is in S.
        """
        if mask & (1 << (x - 1)):
            return True

        # Case 1: x acts as a member 'a' in a triple (a,b,c)
        # We need to ensure there are no distinct b, c in S such that
        # x+b in S, x+c in S, and b+c in S.
        
        # U_x = { y in S | x+y in S }
        U_x = mask & (mask >> x)
        
        if U_x == 0:
            # No b such that x+b in S, so Case 1 is automatically satisfied
            pass
        else:
            # Iterate through bits in U_x. These are potential b's.
            m = U_x
            while m:
                lsb_b = m & -m
                b_idx = lsb_b.bit_length() - 1
                b_val = b_idx + 1
                m ^= lsb_b
                
                # For this b, we need to find c in U_x, c != b, such that b+c in S.
                # V_b = { c in S | b+c in S } = mask & (mask >> b_val)
                V_b = mask & (mask >> b_val)
                
                # We need c in U_x and c in V_b and c != b.
                common = U_x & V_b
                if common & ~lsb_b:
                    return False

        # Case 2: x acts as a sum a+b = x, with a,b in S.
        # We need to ensure there is no c in S \ {a,b} such that a+c in S and b+c in S.
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            
            # V_a = { c in S | a+c in S }
            V_a = mask & (mask >> a)
            # V_b = { c in S | b+c in S }
            V_b = mask & (mask >> b)
            
            common = V_a & V_b
            # Remove a and b from common
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False

        return True

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    # ---- Improved Structural Start ----
    # Start with high numbers (less likely to form sum conflicts)
    # and carefully add lower numbers
    K = 64
    H = list(range(K, max_val + 1))  # {64..128}
    
    # For lower numbers, be more selective
    # Try adding numbers from 32 down to 1, checking validity
    best_mask = 0
    for v in H:
        best_mask |= (1 << (v - 1))
    
    # Now try adding lower numbers greedily
    for v in range(32, 0, -1):
        if is_valid_addition(v, best_mask):
            best_mask |= (1 << (v - 1))
    
    best_size = popcount(best_mask)

    # Try alternative starting points
    alt_starts = [
        # (K, lower_start, lower_end)
        (64, 32, 1),
        (60, 30, 1),
        (70, 35, 1),
        (64, 20, 1),
        (56, 28, 1),
    ]
    
    for K_alt, low_start, low_end in alt_starts:
        alt_mask = 0
        for v in range(K_alt, max_val + 1):
            alt_mask |= (1 << (v - 1))
        for v in range(low_start, low_end - 1, -1):
            if is_valid_addition(v, alt_mask):
                alt_mask |= (1 << (v - 1))
        alt_size = popcount(alt_mask)
        if alt_size > best_size:
            best_size = alt_size
            best_mask = alt_mask

    current_mask = best_mask
    current_size = best_size
    current_elements = elements_of(current_mask)
    unused_elements = set(unused_of(current_mask))

    no_progress_count = 0
    max_no_progress = 150

    # Precompute a list of all elements for faster iteration
    all_elements = list(range(1, max_val + 1))

    while time.time() < deadline - 2.0:
        # 1. Greedy Add - try all unused elements, pick first valid
        if unused_elements:
            added = False
            # Try in random order for diversity
            candidates = list(unused_elements)
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    unused_elements.remove(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    no_progress_count = 0
                    break
            
            if added:
                continue

        # 2. Multi-swap strategies
        found_swap = False
        
        # 2a. 1-out, 2-in Swap
        if current_elements and len(unused_elements) >= 2:
            x_out = random.choice(current_elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            valid_adds = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_adds.append(x)
            
            if len(valid_adds) >= 2:
                random.shuffle(valid_adds)
                max_checks = 30
                checks = 0
                for i in range(len(valid_adds)):
                    if checks > max_checks: break
                    x1 = valid_adds[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_adds)):
                        if checks > max_checks: break
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
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            
                            found_swap = True
                            no_progress_count = 0
                            break
                    if found_swap:
                        break

        # 2b. 2-out, 3-in Swap (if 1-out-2-in didn't work)
        if not found_swap and current_elements and len(unused_elements) >= 3:
            # Try removing 2 random elements and adding 3
            if len(current_elements) >= 2:
                out_indices = random.sample(range(len(current_elements)), 2)
                x_out1 = current_elements[out_indices[0]]
                x_out2 = current_elements[out_indices[1]]
                
                mask_trial = current_mask & ~(1 << (x_out1 - 1)) & ~(1 << (x_out2 - 1))
                
                valid_adds = []
                for x in unused_elements:
                    if is_valid_addition(x, mask_trial):
                        valid_adds.append(x)
                
                if len(valid_adds) >= 3:
                    random.shuffle(valid_adds)
                    max_checks = 20
                    checks = 0
                    for i in range(len(valid_adds)):
                        if checks > max_checks: break
                        x1 = valid_adds[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_adds)):
                            if checks > max_checks: break
                            x2 = valid_adds[j]
                            mask_2 = mask_1 | (1 << (x2 - 1))
                            for k in range(j + 1, len(valid_adds)):
                                if checks > max_checks: break
                                checks += 1
                                x3 = valid_adds[k]
                                if is_valid_addition(x3, mask_2):
                                    current_mask = mask_2 | (1 << (x3 - 1))
                                    current_size += 1
                                    current_elements.remove(x_out1)
                                    current_elements.remove(x_out2)
                                    current_elements.append(x1)
                                    current_elements.append(x2)
                                    current_elements.append(x3)
                                    unused_elements.remove(x1)
                                    unused_elements.remove(x2)
                                    unused_elements.remove(x3)
                                    unused_elements.add(x_out1)
                                    unused_elements.add(x_out2)
                                    
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_mask = current_mask
                                    
                                    found_swap = True
                                    no_progress_count = 0
                                    break
                            if found_swap:
                                break
                        if found_swap:
                            break

        if found_swap:
            continue

        # 3. Perturbation - remove 1-3 random elements
        if current_elements:
            num_remove = random.randint(1, 3)
            num_remove = min(num_remove, len(current_elements))
            for _ in range(num_remove):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
                    unused_elements.add(x_out)
            no_progress_count += 1

        if no_progress_count > max_no_progress:
            # Restart from best with some randomization
            current_mask = best_mask
            current_size = best_size
            current_elements = elements_of(current_mask)
            unused_elements = set(unused_of(current_mask))
            no_progress_count = 0
            
            # Try a few random removals to escape local optimum
            for _ in range(random.randint(1, 5)):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
                    unused_elements.add(x_out)

    # Final polish - multiple passes
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_unused = set(unused_of(final_mask))
        
        # Multiple polish passes with different orderings
        for pass_num in range(5):
            if time.time() > deadline - 1.0:
                break
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                # Alternate between sorted and random order
                if pass_num % 2 == 0:
                    candidates = sorted(final_unused)
                else:
                    candidates = list(final_unused)
                    random.shuffle(candidates)
                for x in candidates:
                    if x not in final_unused:
                        continue
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
