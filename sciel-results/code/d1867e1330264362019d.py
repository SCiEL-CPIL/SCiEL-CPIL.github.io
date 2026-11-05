# EVOLVE-BLOCK-START
import math
import bisect

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i] = L / (i+1) for i in 0..29
    w = [L // (i + 1) for i in range(30)]
    
    n1 = 15
    n2 = 15
    
    # Pre-compute masks for second half
    masks_plus2 = [1 << (2 * (15 + i)) for i in range(15)]
    masks_minus2 = [2 << (2 * (15 + i)) for i in range(15)]
    
    # Generate all (sum, state) for the second half (indices 15..29)
    # Iterative generation
    list2 = [(0, 0)]
    for i in range(15):
        w_idx = w[15 + i]
        m_plus = masks_plus2[i]
        m_minus = masks_minus2[i]
        new_list2 = []
        append = new_list2.append
        for s, st in list2:
            append((s, st))
            append((s + w_idx, st | m_plus))
            append((s - w_idx, st | m_minus))
        list2 = new_list2
    
    # Sort list2 by sum
    list2.sort(key=lambda x: x[0])
    
    # Extract sums for bisect
    sums2 = [s for s, _ in list2]
    
    best_abs_T = float('inf')
    best_state = 0
    
    # Pre-compute masks for first half
    masks_plus1 = [1 << (2 * i) for i in range(15)]
    masks_minus1 = [2 << (2 * i) for i in range(15)]
    
    # Iterative search for first half
    # Stack items: (idx, current_sum, current_state)
    stack = [(0, 0, 0)]
    found_optimal = False
    
    while stack and not found_optimal:
        idx, c_sum, c_state = stack.pop()
        
        if idx == 15:
            # Find s2 in sums2 closest to -current_sum
            target = -c_sum
            pos = bisect.bisect_left(sums2, target)
            
            # Check pos and pos-1
            if pos < len(sums2):
                s2, state2 = list2[pos]
                total_sum = c_sum + s2
                if total_sum != 0:
                    abs_t = abs(total_sum)
                    if abs_t < best_abs_T:
                        best_abs_T = abs_t
                        best_state = c_state | state2
                        if best_abs_T == 1:
                            found_optimal = True
                            break
            if not found_optimal and pos > 0:
                s2, state2 = list2[pos - 1]
                total_sum = c_sum + s2
                if total_sum != 0:
                    abs_t = abs(total_sum)
                    if abs_t < best_abs_T:
                        best_abs_T = abs_t
                        best_state = c_state | state2
                        if best_abs_T == 1:
                            found_optimal = True
                            break
        else:
            w_idx = w[idx]
            m_plus = masks_plus1[idx]
            m_minus = masks_minus1[idx]
            
            # Push children
            # Order: 0, +1, -1
            stack.append((idx + 1, c_sum, c_state))
            stack.append((idx + 1, c_sum + w_idx, c_state | m_plus))
            stack.append((idx + 1, c_sum - w_idx, c_state | m_minus))
            
    # Construct coefficients
    coeffs = [0] * 30
    for i in range(30):
        val = (best_state >> (2 * i)) & 3
        if val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
        else:
            coeffs[i] = 0
            
    return coeffs
# EVOLVE-BLOCK-END
