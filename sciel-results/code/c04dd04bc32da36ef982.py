import random
import time
import math

def construct():
    N = 256
    
    # Precompute 3-APs and 4-APs as bitmasks
    # Bit i corresponds to element i+1 (1-indexed)
    # Mask for element x is 1 << (x-1)
    
    three_aps = []
    four_aps = []
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
            
    # Precompute for each element, the masks of 4-APs it belongs to
    # This allows fast validity check: 
    # Adding x is valid if no 4-AP mask M containing x has (M & ~x_mask) subset of current_mask
    # i.e., (M & current_mask) == (M & ~x_mask)
    
    elem_to_fours = [[] for _ in range(N + 1)]
    for mask in four_aps:
        # Find elements in mask
        # Since N=256, we can iterate bits or just check 1..N
        # Faster: iterate d and a again? No, just use the mask.
        # For each bit set in mask, add mask to elem_to_fours[bit]
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() # 1-indexed position
            elem_to_fours[idx].append(mask)
            m ^= lsb
            
    # Function to calculate score (number of 3-APs)
    def calc_score(mask):
        count = 0
        for ap_mask in three_aps:
            if (mask & ap_mask) == ap_mask:
                count += 1
        return count
        
    # Function to check if adding element x to mask is valid
    def can_add(mask, x):
        x_mask = 1 << (x - 1)
        # Check all 4-APs containing x
        for ap_mask in elem_to_fours[x]:
            # The other 3 elements
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                return False
        return True
        
    # Greedy Repair: Start with full set, remove elements to break 4-APs
    def greedy_repair():
        mask = (1 << N) - 1 # All elements 1..256
        # We want to remove elements to break all 4-APs
        # Strategy: While there is a 4-AP, remove an element from it
        # To maximize 3-APs, remove the element that is in the fewest 3-APs?
        # Or just remove randomly to break 4-APs, then optimize.
        
        # Better: Iterate through 4-APs. If present, remove one element.
        # Which one? The one with the highest "degree" in 4-APs? 
        # Removing a high-degree element might break more 4-APs at once.
        
        # Let's try: 
        # 1. Find all 4-APs present.
        # 2. Count occurrences of each element in these 4-APs.
        # 3. Remove the element with the highest count.
        # 4. Repeat.
        
        # This is $O(N \times |4APs|)$ per step? 
        # |4APs| ~ 12k. N=256. Steps ~ ?
        # Max 256 removals.
        # 256 * 12k = 3M ops. Fast enough.
        
        while True:
            # Find active 4-APs
            active_fours = []
            for ap_mask in four_aps:
                if (mask & ap_mask) == ap_mask:
                    active_fours.append(ap_mask)
                    
            if not active_fours:
                break
                
            # Count element frequencies in active_fours
            freq = [0] * (N + 1)
            for ap_mask in active_fours:
                m = ap_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length()
                    freq[idx] += 1
                    m ^= lsb
                    
            # Pick element with max freq
            max_freq = -1
            rem_elem = -1
            for i in range(1, N + 1):
                if freq[i] > max_freq:
                    max_freq = freq[i]
                    rem_elem = i
                    
            # Remove it
            if rem_elem != -1:
                mask &= ~(1 << (rem_elem - 1))
                
        return mask

    # Local Search
    def local_search(initial_mask, time_limit):
        current_mask = initial_mask
        current_score = calc_score(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        start_time = time.time()
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        # Precompute list of elements for random selection
        elements = list(range(1, N + 1))
        
        while T > T_min and time.time() - start_time < time_limit:
            x = random.choice(elements)
            x_mask = 1 << (x - 1)
            
            if current_mask & x_mask:
                # Remove x
                new_mask = current_mask & ~x_mask
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # Add x
                if can_add(current_mask, x):
                    new_mask = current_mask | x_mask
                    new_score = calc_score(new_score_mask if False else new_mask) # Typo fix: new_mask
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                # else: invalid move, ignore
                
            T *= cooling
            
        return best_mask

    # Run multiple greedy repairs to find a good start
    best_initial_mask = 0
    best_initial_score = -1
    
    # Run greedy repair a few times with randomization in tie-breaking?
    # Or just once, it's deterministic mostly.
    # Let's run it once.
    init_mask = greedy_repair()
    init_score = calc_score(init_mask)
    
    # To improve diversity, we can randomize the choice of element to remove in greedy_repair
    # Let's modify greedy_repair to accept a random seed or just pick randomly among max freq
    
    def greedy_repair_random():
        mask = (1 << N) - 1
        while True:
            active_fours = []
            for ap_mask in four_aps:
                if (mask & ap_mask) == ap_mask:
                    active_fours.append(ap_mask)
            if not active_fours:
                break
            freq = [0] * (N + 1)
            for ap_mask in active_fours:
                m = ap_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length()
                    freq[idx] += 1
                    m ^= lsb
            max_freq = max(freq)
            candidates = [i for i in range(1, N + 1) if freq[i] == max_freq]
            rem_elem = random.choice(candidates)
            mask &= ~(1 << (rem_elem - 1))
        return mask

    best_init_mask = 0
    best_init_score = -1
    time_start = time.time()
    
    # Try a few random greedy repairs
    for _ in range(5):
        if time.time() - time_start > 5:
            break
        m = greedy_repair_random()
        s = calc_score(m)
        if s > best_init_score:
            best_init_score = s
            best_init_mask = m
            
    # Run local search on the best initial mask
    final_mask = local_search(best_init_mask, 50)
    
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if final_mask & (1 << (i - 1)):
            result.append(i)
            
    return sorted(result)
