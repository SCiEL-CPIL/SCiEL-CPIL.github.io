"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 128
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
        # We need to ensure there are no distinct b, c in S such that
        # x+b in S, x+c in S, and b+c in S.
        
        # U_x = { y in S | x+y in S }
        U_x = mask & (mask >> x)
        
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

    def popcount(x):
        return bin(x).count('1')

    # ---- Structural start ----
    # Start with the known good structure: L U H
    K = 64
    H = list(range(K, max_val + 1))                 # {64..128}
    max_L = (K - 1) // 2                            # 31
    start_L = max_L // 2 + 1                        # 16
    L = list(range(start_L, max_L + 1))             # {16..31}

    best_mask = 0
    for v in H + L:
        best_mask |= (1 << (v - 1))
    best_size = popcount(best_mask)

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    # Try multiple random restarts with different initial structures
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break

        if restart == 0:
            # Use the structural initial solution
            current_mask = best_mask
        else:
            # Try different K values or random subsets
            random.seed(restart * 1000)
            current_mask = 0
            # Start with a random subset of the upper half
            for v in range(32, max_val + 1):
                if random.random() < 0.5:
                    current_mask |= (1 << (v - 1))
            # Add some from lower half
            for v in range(1, 32):
                if random.random() < 0.3:
                    current_mask |= (1 << (v - 1))

        current_size = popcount(current_mask)
        current_elements = elements_of(current_mask)
        unused_elements = set(unused_of(current_mask))

        no_progress_count = 0
        max_no_progress = 50

        while time.time() < deadline - 3.0:
            # 1. Greedy Add
            if unused_elements:
                added = False
                candidates = list(unused_elements)
                random.shuffle(candidates)
                for x in candidates[:15]:
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
                    max_checks = 25
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
                                
                                if current_size > best_size:
                                    best_size = current_size
                                    best_mask = current_mask
                                
                                found_swap = True
                                no_progress_count = 0
                                break
                        if found_swap:
                            break
                    
                    if found_swap:
                        continue

            # 3. Perturbation
            if current_elements:
                x_out = random.choice(current_elements)
                current_mask &= ~(1 << (x_out - 1))
                current_size -= 1
                current_elements.remove(x_out)
                unused_elements.add(x_out)
                no_progress_count += 1

            if no_progress_count > max_no_progress:
                # Reset to best found so far
                current_mask = best_mask
                current_size = best_size
                current_elements = elements_of(current_mask)
                unused_elements = set(unused_of(current_mask))
                no_progress_count = 0

    # Final polish
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_unused = set(unused_of(final_mask))
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for x in list(final_unused):
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

# EVOLVE-BLOCK-END
