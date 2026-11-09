import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512
    
    def time_ok():
        return time.time() < deadline

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def init_ap_count(mask):
        """Initialize ap_count array and return score."""
        ap_count = [0] * (MAX_D + 1)
        score = 0
        for d in range(1, MAX_D + 1):
            # Count number of 3-APs with difference d
            # a, a+d, a+2d in mask
            # Iterate a from 1 to MAX_VAL - 2*d
            count = 0
            # Using bit operations might be faster, but loop is clear
            # For speed, we can check:
            # (mask & (mask >> d) & (mask >> (2*d)))
            # This gives a mask of all 'a' that start an AP.
            # The number of such 'a' is popcount of that mask.
            # But we only care if count > 0.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count = 1 # We only need to know if > 0 for score, 
                          # but for delta, we need exact count?
                # If we swap, we need to know if count goes from 1 to 0 or 2 to 1.
                # So we need exact count.
                # Let's calculate exact count.
                m_ap = mask & (mask >> d) & (mask >> (2 * d))
                # Count bits in m_ap
                # Python 3.8+ has int.bit_count()
                count = m_ap.bit_count()
                if count > 0:
                    score += 1
            ap_count[d] = count
        return ap_count, score

    def update_remove(v, mask, ap_count, score):
        """Remove v from mask, update ap_count and score. Returns new mask, ap_count, score."""
        # v is 1-indexed
        # Check all d where v could be part of an AP
        # We iterate d and check if v is in an AP for that d
        
        # v as first term: a=v. Need v+d, v+2d in mask.
        max_d1 = (MAX_VAL - v) // 2
        for d in range(1, max_d1 + 1):
            if (mask >> (v + d - 1)) & 1 and (mask >> (v + 2*d - 1)) & 1:
                ap_count[d] -= 1
                if ap_count[d] == 0:
                    score -= 1
        
        # v as middle term: a+d=v => a=v-d. Need v-d, v+d in mask.
        max_d2 = min(v - 1, MAX_VAL - v)
        for d in range(1, max_d2 + 1):
            if (mask >> (v - d - 1)) & 1 and (mask >> (v + d - 1)) & 1:
                ap_count[d] -= 1
                if ap_count[d] == 0:
                    score -= 1
        
        # v as last term: a+2d=v => a=v-2d. Need v-2d, v-d in mask.
        max_d3 = (v - 1) // 2
        for d in range(1, max_d3 + 1):
            if (mask >> (v - 2*d - 1)) & 1 and (mask >> (v - d - 1)) & 1:
                ap_count[d] -= 1
                if ap_count[d] == 0:
                    score -= 1
        
        mask &= ~(1 << (v - 1))
        return mask, ap_count, score

    def update_add(v, mask, ap_count, score):
        """Add v to mask, update ap_count and score. Returns new mask, ap_count, score."""
        mask |= 1 << (v - 1)
        
        # v as first term
        max_d1 = (MAX_VAL - v) // 2
        for d in range(1, max_d1 + 1):
            if (mask >> (v + d - 1)) & 1 and (mask >> (v + 2*d - 1)) & 1:
                ap_count[d] += 1
                if ap_count[d] == 1:
                    score += 1
        
        # v as middle term
        max_d2 = min(v - 1, MAX_VAL - v)
        for d in range(1, max_d2 + 1):
            if (mask >> (v - d - 1)) & 1 and (mask >> (v + d - 1)) & 1:
                ap_count[d] += 1
                if ap_count[d] == 1:
                    score += 1
        
        # v as last term
        max_d3 = (v - 1) // 2
        for d in range(1, max_d3 + 1):
            if (mask >> (v - 2*d - 1)) & 1 and (mask >> (v - d - 1)) & 1:
                ap_count[d] += 1
                if ap_count[d] == 1:
                    score += 1
        
        return mask, ap_count, score

    def get_targeted_moves(mask, s_set, current_list):
        """Generate candidate new values that would complete 3-APs with existing elements."""
        candidates = set()
        lst_set = set(current_list)
        
        for i in range(len(current_list)):
            for j in range(len(current_list)):
                if i == j:
                    continue
                a = current_list[i]
                b = current_list[j]
                # If a and b are endpoints of a 3-AP, middle = (a+b)/2
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if 1 <= mid <= MAX_VAL and mid not in lst_set:
                        candidates.add(mid)
                # If a and b are first two terms, third = 2*b - a
                third = 2 * b - a
                if 1 <= third <= MAX_VAL and third not in lst_set:
                    candidates.add(third)
                # If a and b are last two terms, first = 2*a - b
                first = 2 * a - b
                if 1 <= first <= MAX_VAL and first not in lst_set:
                    candidates.add(first)
        
        return list(candidates)

    def local_search(init_mask, init_score, init_ap_count, init_list, time_limit):
        """
        Performs SA with fast delta updates.
        Returns best_mask, best_score, best_ap_count, best_list
        """
        current_mask = init_mask
        current_score = init_score
        current_ap_count = init_ap_count[:] # Copy
        current_list = init_list[:]
        current_set = set(current_list)
        
        best_mask = current_mask
        best_score = current_score
        best_ap_count = current_ap_count[:]
        best_list = current_list[:]
        
        T = 10.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Mix of random and targeted moves
            use_targeted = (random.random() < 0.3)
            
            if use_targeted:
                targeted = get_targeted_moves(current_mask, current_set, current_list)
                if targeted:
                    new_val = random.choice(targeted)
                else:
                    new_val = random.randint(1, MAX_VAL)
            else:
                new_val = random.randint(1, MAX_VAL)
            
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            
            if new_val == old_val or new_val in current_set:
                T *= cooling_rate
                continue
            
            # Calculate delta by simulating remove and add
            # We need to be careful with ap_count updates
            
            # Temporarily store state
            old_ap_count = current_ap_count[:]
            old_score = current_score
            old_mask = current_mask
            
            # Remove old_val
            temp_mask, temp_ap_count, temp_score = update_remove(old_val, current_mask, current_ap_count, current_score)
            
            # Add new_val
            temp_mask, temp_ap_count, temp_score = update_add(new_val, temp_mask, temp_ap_count, temp_score)
            
            delta = temp_score - old_score
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept move
                current_mask = temp_mask
                current_score = temp_score
                current_ap_count = temp_ap_count
                current_set.discard(old_val)
                current_set.add(new_val)
                current_list[idx] = new_val
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_ap_count = current_ap_count[:]
                    best_list = current_list[:]
            
            T *= cooling_rate
            
        return best_mask, best_score, best_ap_count, best_list

    random.seed(42)
    
    best_mask = 0
    best_score = 0
    best_list = []
    best_ap_count = None
    
    # Population of size 4
    population = []
    
    # Initial diverse starts
    starts = []
    
    # 1. Greedy
    mask_g = 0
    set_g = set()
    for _ in range(N):
        best_new = None
        best_s = -1
        # Sample candidates
        candidates = random.sample(range(1, MAX_VAL + 1), 200)
        for v in candidates:
            if v in set_g:
                continue
            new_mask = mask_g | (1 << (v - 1))
            # Fast score estimate for greedy? Just use full score for simplicity in init
            # Or use ap_count init?
            # For init, full score is fine (32 * 255 ops)
            s = 0
            for d in range(1, MAX_D + 1):
                if (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0:
                    s += 1
            if s > best_s:
                best_s = s
                best_new = v
        if best_new is None:
            best_new = random.choice([v for v in range(1, MAX_VAL + 1) if v not in set_g])
        mask_g |= (1 << (best_new - 1))
        set_g.add(best_new)
    
    ap_count_g, score_g = init_ap_count(mask_g)
    starts.append((mask_g, score_g, ap_count_g, mask_to_list(mask_g)))
    
    # 2. Interval
    lst_i = list(range(1, 33))
    mask_i = list_to_mask(lst_i)
    ap_count_i, score_i = init_ap_count(mask_i)
    starts.append((mask_i, score_i, ap_count_i, lst_i))
    
    # 3. AP
    lst_a = [1 + i * 16 for i in range(N)]
    mask_a = list_to_mask(lst_a)
    ap_count_a, score_a = init_ap_count(mask_a)
    starts.append((mask_a, score_a, ap_count_a, lst_a))
    
    # 4. Random
    lst_r = random.sample(range(1, MAX_VAL + 1), N)
    mask_r = list_to_mask(lst_r)
    ap_count_r, score_r = init_ap_count(mask_r)
    starts.append((mask_r, score_r, ap_count_r, lst_r))
    
    # Run local search on each start
    for i, (m, s, ac, l) in enumerate(starts):
        if not time_ok():
            break
        # Allocate time for each
        t_limit = time.time() + (140.0 / 4) # ~35s each
        bm, bs, bac, bl = local_search(m, s, ac, l, t_limit)
        
        if bs > best_score:
            best_score = bs
            best_mask = bm
            best_ap_count = bac
            best_list = bl[:]
            
        print(f"Start {i}: {bs}", file=sys.stderr)
        
    # If time remains, do more rounds with perturbation of best
    if time_ok():
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Perturb best solution
            perturbed_mask = best_mask
            perturbed_score = best_score
            perturbed_ap_count = best_ap_count[:]
            perturbed_list = best_list[:]
            
            # Randomly swap 2 elements
            for _ in range(2):
                idx1 = random.randint(0, N-1)
                idx2 = random.randint(0, N-1)
                if idx1 != idx2:
                    v1 = perturbed_list[idx1]
                    v2 = perturbed_list[idx2]
                    # Swap in mask
                    # Remove v1, Add v2 to idx1? No, just swap values in list and mask
                    # Easier: Remove both, Add both at swapped positions?
                    # Actually, just swap values in list, and update mask
                    # Mask update:
                    perturbed_mask &= ~(1 << (v1 - 1))
                    perturbed_mask &= ~(1 << (v2 - 1))
                    perturbed_mask |= 1 << (v1 - 1) # Wait, v1 is still there, just moved?
                    # No, the set is the same. The mask is the same.
                    # The list order doesn't matter for the set.
                    # So "perturbation" should be changing the set.
                    # Let's replace 2 random elements with new random ones.
                    new_v1 = random.randint(1, MAX_VAL)
                    new_v2 = random.randint(1, MAX_VAL)
                    
                    # Ensure distinct
                    while new_v1 == new_v2 or new_v1 in perturbed_list or new_v2 in perturbed_list:
                         new_v1 = random.randint(1, MAX_VAL)
                         new_v2 = random.randint(1, MAX_VAL)
                    
                    # Update
                    perturbed_mask, perturbed_ap_count, perturbed_score = update_remove(v1, perturbed_mask, perturbed_ap_count, perturbed_score)
                    perturbed_list[idx1] = new_v1
                    perturbed_mask, perturbed_ap_count, perturbed_score = update_add(new_v1, perturbed_mask, perturbed_ap_count, perturbed_score)
                    
                    perturbed_mask, perturbed_ap_count, perturbed_score = update_remove(v2, perturbed_mask, perturbed_ap_count, perturbed_score)
                    perturbed_list[idx2] = new_v2
                    perturbed_mask, perturbed_ap_count, perturbed_score = update_add(new_v2, perturbed_mask, perturbed_ap_count, perturbed_score)
            
            t_limit = time.time() + (remaining_time * 0.8)
            bm, bs, bac, bl = local_search(perturbed_mask, perturbed_score, perturbed_ap_count, perturbed_list, t_limit)
            
            if bs > best_score:
                best_score = bs
                best_mask = bm
                best_ap_count = bac
                best_list = bl[:]
            
            print(f"Perturbed Best: {bs}", file=sys.stderr)

    print(f"Final Score: {best_score}", file=sys.stderr)
    return best_list

if __name__ == "__main__":
    construct()
