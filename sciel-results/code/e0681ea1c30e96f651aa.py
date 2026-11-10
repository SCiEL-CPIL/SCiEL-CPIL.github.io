import time
import random
import math

N = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _precompute():
    """
    Precompute 3-APs and 4-APs with bitmask representations for fast operations.
    """
    aps3 = []
    aps4 = []
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            mask_3 = (1 << a) | (1 << b) | (1 << c)
            # For delta calculation, we need to know which other elements form a 3-AP with x
            # If x is in the AP, the other two are the "partners"
            elem_to_aps3[a].append((1 << b) | (1 << c))
            elem_to_aps3[b].append((1 << a) | (1 << c))
            elem_to_aps3[c].append((1 << a) | (1 << b))
            
    # Generate 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            # For validity check: if we add y, do we complete a 4-AP?
            # This happens if the other 3 elements of a 4-AP containing y are already in A.
            elem_to_aps4[a].append((1 << b) | (1 << c) | (1 << e))
            elem_to_aps4[b].append((1 << a) | (1 << c) | (1 << e))
            elem_to_aps4[c].append((1 << a) | (1 << b) | (1 << e))
            elem_to_aps4[e].append((1 << a) | (1 << b) | (1 << c))
    
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_mask(elem_to_aps3, elem_to_aps4):
    """
    Greedy construction: add x if it doesn't complete a 4-AP.
    Prioritize elements with more 3-AP connections.
    """
    A_mask = 0
    ap_count = [len(elem_to_aps3[i]) for i in range(N + 1)]
    candidates = sorted(range(1, N + 1), key=lambda x: -ap_count[x])
    
    for x in candidates:
        valid = True
        for mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
    return A_mask

def construct():
    start_time = time.time()
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute()
    
    def calc_score(mask):
        score = 0
        for a, b, c in aps3:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                score += 1
        return score

    # Initial greedy set
    A_mask = _initial_greedy_mask(elem_to_aps3, elem_to_aps4)
    current_score = calc_score(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    # Helper: check if adding y to mask (with x removed) creates a 4-AP
    def is_valid_swap(x, y, mask):
        # We remove x and add y.
        # A 4-AP is formed if y is in a 4-AP (y, u, v, w) and u,v,w are in mask \ {x}.
        # Note: x cannot be in the 4-AP with y if x is removed? 
        # Wait, the 4-AP must be *entirely* in the new set.
        # New set = (Old Set - {x}) + {y}.
        # So we check all 4-APs containing y.
        # For each 4-AP (y, u, v, w), we need {u, v, w} subset of (Old Set - {x}).
        # This is equivalent to: {u, v, w} subset of Old Set AND x is NOT in {u, v, w}.
        # Since x != y, and the 4-AP is distinct elements, x is either in the 4-AP or not.
        # If x is in the 4-AP, then one of u,v,w is x.
        # So we check: (mask & mask_other3) == mask_other3.
        # If this is true, it means u,v,w are in mask.
        # If x is one of u,v,w, then after removing x, the set is not complete, so it's valid.
        # If x is NOT one of u,v,w, then after removing x (which isn't there), the set is still complete, so it's INVALID.
        
        for mask_other3 in elem_to_aps4[y]:
            if (mask & mask_other3) == mask_other3:
                # The other 3 elements are in the current mask.
                # If x is one of them, removing x breaks the AP.
                # If x is not one of them, the AP remains.
                # mask_other3 contains bits for u,v,w.
                if (mask_other3 & (1 << x)) == 0:
                    return False
        return True

    def calc_delta_swap(x, y, mask):
        """
        Calculate change in number of 3-APs when removing x and adding y.
        Loss: 3-APs containing x where other 2 are in mask.
        Gain: 3-APs containing y where other 2 are in mask (and x is not one of the other 2, which is true since x!=y).
        Wait, if a 3-AP is (x, u, v), and we remove x, we lose it if u,v in mask.
        If a 3-AP is (y, u, v), and we add y, we gain it if u,v in mask.
        Note: If a 3-AP contains both x and y, e.g., (x, y, z), 
        Loss: (x,y,z) has other 2 (y,z). y is in mask. So we lose it.
        Gain: (x,y,z) has other 2 (x,z). x is NOT in mask (we are removing it). So we don't gain it.
        This is correct.
        """
        loss = 0
        gain = 0
        
        for mask_other2 in elem_to_aps3[x]:
            if (mask & mask_other2) == mask_other2:
                loss += 1
                
        for mask_other2 in elem_to_aps3[y]:
            # We need the other 2 to be in the NEW mask.
            # New mask = mask - {x} + {y}.
            # The other 2 elements are in mask_other2.
            # They must be in mask, AND they must not be x (which is true as x!=y and mask_other2 doesn't contain y).
            # Actually, mask_other2 contains the other 2 elements.
            # If x is one of the other 2 elements? No, x is the center/end, the others are distinct.
            # So we just check if they are in mask.
            # Wait, if x is one of the "other 2", then (mask & mask_other2) would include x.
            # But we are removing x.
            # So we should check if the other 2 are in (mask - {x}).
            # This is equivalent to: (mask & mask_other2) == mask_other2 AND (mask_other2 & (1<<x)) == 0.
            # Since x is not in the AP with y as the focal point? 
            # elem_to_aps3[y] stores masks of the OTHER two elements.
            # So mask_other2 does NOT contain y.
            # Does it contain x? Yes, if x, y are in a 3-AP.
            # If x is in mask_other2, then (mask & mask_other2) == mask_other2 implies x is in mask.
            # But we are removing x. So the AP is not formed in the new set.
            # So we must exclude cases where x is in mask_other2.
            
            if (mask & mask_other2) == mask_other2:
                if (mask_other2 & (1 << x)) == 0:
                    gain += 1
        return gain - loss

    # Local Search Loop
    # We use a simple simulated annealing / random walk with hill climbing bias
    T = 5.0
    cooling_rate = 0.999
    
    while (time.time() - start_time) < DEADLINE:
        # Get in/out sets
        in_set = [i for i in range(1, N + 1) if (A_mask >> i) & 1]
        out_set = [i for i in range(1, N + 1) if not ((A_mask >> i) & 1)]
        
        if not in_set or not out_set:
            break
            
        # Pick random swap
        x = random.choice(in_set)
        y = random.choice(out_set)
        
        if not is_valid_swap(x, y, A_mask):
            continue
            
        delta = calc_delta_swap(x, y, A_mask)
        
        # Acceptance
        accept = False
        if delta >= 0:
            accept = True
        else:
            # exp(delta / T)
            # delta is negative
            if T > 0.01:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    accept = True
            # else reject
            
        if accept:
            A_mask &= ~(1 << x)
            A_mask |= (1 << y)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
                
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return sorted([i for i in range(1, N + 1) if (best_mask >> i) & 1])
