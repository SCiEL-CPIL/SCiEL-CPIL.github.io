import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 60s, but we want to return well before.
    deadline = start_time + 55.0
    
    n = P['n']
    max_val = n
    full_mask = (1 << n) - 1
    
    # Precompute pairs for sums
    # pairs_sum[s] = list of (a, b) such that a + b = s, a < b
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))
    
    # Precompute for each value x, the bitmask of values b such that x+b <= max_val
    # and x+b is a valid sum partner. This helps speed up case 1.
    
    def get_bits(mask):
        """Yield indices of set bits."""
        m = mask
        while m:
            lsb = m & -m
            yield lsb.bit_length() - 1
            m ^= lsb

    def is_valid_addition(x, mask):
        """
        Check if adding x to valid set 'mask' results in a valid set.
        x is 1-indexed. mask is bitset where bit i corresponds to value i+1.
        """
        if mask & (1 << (x - 1)):
            return True # Already in set
            
        # Case 1: x is one of the members a,b,c.
        # WLOG a=x. Need b,c in S such that b+c in S, x+b in S, x+c in S.
        # Let A = {b in S | x+b in S}
        # We need to check if there exist distinct b, c in A such that b+c in S.
        
        # Efficiently find A using bitmask operations
        # b is in A if bit (b-1) is set in mask AND bit (x+b-1) is set in mask.
        # For each b in 1..max_val-x, if b is in S and x+b is in S, then b is in A.
        
        # Build A_bits: for b from 1 to max_val-x, check if both b and x+b are in mask
        A_bits = 0
        for b in range(1, max_val - x + 1):
            b_bit = 1 << (b - 1)
            sum_bit = 1 << (x + b - 1)
            if (mask & b_bit) and (mask & sum_bit):
                A_bits |= b_bit
        
        # Now check if there are b, c in A_bits, b != c, such that b+c in S
        # Iterate b in A_bits
        m = A_bits
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            
            # We need c in A_bits, c != b, such that b+c in S
            # For each c in A_bits where c != b, check if b+c is in mask
            m_c = A_bits & ~lsb_b  # A_bits without b
            while m_c:
                lsb_c = m_c & -m_c
                c_idx = lsb_c.bit_length() - 1
                c_val = c_idx + 1
                m_c ^= lsb_c
                
                sum_val = b_val + c_val
                if sum_val <= max_val and (mask & (1 << (sum_val - 1))):
                    return False
        
        # Case 2: x is one of the sums.
        # WLOG a+b = x. a,b in S.
        # Need c in S, c != a, c != b, such that a+c in S and b+c in S.
        
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
                
            # Found a,b in S summing to x.
            # Check for c
            # We need c in S \ {a,b} such that (a+c in S) and (b+c in S)
            
            # Build U_a = {c in S | a+c in S} and U_b = {c in S | b+c in S}
            U_a = 0
            for c in range(1, max_val - a + 1):
                c_bit = 1 << (c - 1)
                sum_bit = 1 << (a + c - 1)
                if (mask & c_bit) and (mask & sum_bit):
                    U_a |= c_bit
            
            U_b = 0
            for c in range(1, max_val - b + 1):
                c_bit = 1 << (c - 1)
                sum_bit = 1 << (b + c - 1)
                if (mask & c_bit) and (mask & sum_bit):
                    U_b |= c_bit
            
            # Check if (U_a & U_b) has any bit other than a or b
            common = U_a & U_b
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False
                
        return True

    def try_add(mask, x):
        if is_valid_addition(x, mask):
            return mask | (1 << (x - 1))
        return mask

    def try_remove(mask, x):
        # Removing an element from a valid set always results in a valid set.
        return mask & ~(1 << (x - 1))

    # Initial valid set: Upper half
    best_mask = 0
    for i in range(n//2, n):
        best_mask |= (1 << i)
    best_size = n - n//2
    
    current_mask = best_mask
    current_size = best_size
    
    # Local Search
    unused_elements = [i for i in range(1, n + 1) if not (current_mask & (1 << (i-1)))]
    current_elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i-1)))]
    
    # Phase 1: Greedy add - try to add any unused element
    while time.time() < deadline:
        # Try greedy add
        if unused_elements:
            random.shuffle(unused_elements)
            added = False
            for x in unused_elements:
                new_mask = try_add(current_mask, x)
                if new_mask != current_mask:
                    # Success
                    current_mask = new_mask
                    current_size += 1
                    unused_elements.remove(x)
                    current_elements.append(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    break
            if added:
                continue
        
        # If we couldn't add, try 1-out, 2-in
        # Remove a random element, then try to add two
        if current_elements:
            # Pick a random element to remove
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            # Try to add two elements from unused
            valid_x1 = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                random.shuffle(valid_x1)
                max_checks = 100
                checks = 0
                found = False
                for i in range(len(valid_x1)):
                    if checks > max_checks: break
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_x1)):
                        if checks > max_checks: break
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            
                            current_elements.remove(x_out)
                            current_elements.append(x1)
                            current_elements.append(x2)
                            unused_elements.remove(x1)
                            unused_elements.remove(x2)
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                                
                            found = True
                            break
                    if found: break
                
                if found:
                    continue
        
        # If stuck, perturb
        if current_elements:
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            current_mask = current_mask & ~(1 << (x_out - 1))
            current_size -= 1
            current_elements.remove(x_out)
            unused_elements.append(x_out)

    # Phase 2: Final local search pass on best_mask
    # Try to improve best_mask with more aggressive search
    final_mask = best_mask
    final_size = best_size
    final_unused = [i for i in range(1, n + 1) if not (final_mask & (1 << (i-1)))]
    final_elements = [i for i in range(1, n + 1) if (final_mask & (1 << (i-1)))]
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try all possible additions systematically
        progress = True
        while progress and time.time() < deadline - 2:
            progress = False
            for x in list(final_unused):
                if is_valid_addition(x, final_mask):
                    final_mask |= (1 << (x - 1))
                    final_size += 1
                    final_unused.remove(x)
                    final_elements.append(x)
                    if final_size > best_size:
                        best_size = final_size
                        best_mask = final_mask
                    progress = True
        
        # Try 1-out, 2-in moves on best
        if time.time() < deadline - 1:
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                if not final_elements:
                    break
                idx_out = random.randint(0, len(final_elements) - 1)
                x_out = final_elements[idx_out]
                
                mask_trial = final_mask & ~(1 << (x_out - 1))
                
                valid_x1 = []
                for x in final_unused:
                    if is_valid_addition(x, mask_trial):
                        valid_x1.append(x)
                
                if len(valid_x1) >= 2:
                    random.shuffle(valid_x1)
                    for i in range(len(valid_x1)):
                        x1 = valid_x1[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_x1)):
                            x2 = valid_x1[j]
                            if is_valid_addition(x2, mask_1):
                                new_mask = mask_1 | (1 << (x2 - 1))
                                if popcount(new_mask) > best_size:
                                    best_size = popcount(new_mask)
                                    best_mask = new_mask
                                break
                        else:
                            continue
                        break

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)

def popcount(x):
    """Count set bits in integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count
