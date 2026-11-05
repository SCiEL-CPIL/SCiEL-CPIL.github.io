import time
import random
import math

def construct():
    N = 512
    # Elements are 1..N, mapped to bits 0..N-1
    # Bit i represents element i+1
    
    # Precompute triples for each element
    # triples_for[x] is a list of integer masks for the other 3 elements
    triples_for = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    # 1 <= a, a+3d <= N
    # d >= 1
    
    # We iterate over d and a
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-indexed bits
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            # For each element in the AP, the other 3 form a triple
            # Element a (index a-1)
            mask_bce = b_bit | c_bit | e_bit
            triples_for[a - 1].append(mask_bce)
            
            # Element b (index b-1)
            mask_ace = a_bit | c_bit | e_bit
            triples_for[b - 1].append(mask_ace)
            
            # Element c (index c-1)
            mask_abe = a_bit | b_bit | e_bit
            triples_for[c - 1].append(mask_abe)
            
            # Element e (index e-1)
            mask_abc = a_bit | b_bit | c_bit
            triples_for[e - 1].append(mask_abc)

    best_S = 0
    best_size = 0
    
    start_time = time.time()
    deadline = 55.0 # Leave 5s buffer for 60s limit
    
    # Helper function to check validity and size
    def get_size(S):
        return bin(S).count('1')
    
    # Helper to greedily fill remaining spots from a given state
    # This is a "completion" heuristic
    def complete_greedily(S):
        size = bin(S).count('1')
        # Try adding elements in random order if not already in S
        # We do a simple pass to add any element that doesn't create an AP
        # We repeat this a few times or until no more can be added
        # For speed, we just do one pass in random order
        candidates = [i for i in range(N) if not (S & (1 << i))]
        random.shuffle(candidates)
        for x_idx in candidates:
            x_bit = 1 << x_idx
            creates_ap = False
            for mask in triples_for[x_idx]:
                if (S & mask) == mask:
                    creates_ap = True
                    break
            if not creates_ap:
                S |= x_bit
                size += 1
        return S, size

    # Multi-start Simulated Annealing + Local Search
    # We run multiple short annealing sessions
    
    # Initial Greedy Construction
    S = 0
    # Randomize order for diversity in first start
    order = list(range(N))
    random.shuffle(order)
    for x_idx in order:
        x_bit = 1 << x_idx
        creates_ap = False
        for mask in triples_for[x_idx]:
            if (S & mask) == mask:
                creates_ap = True
                break
        if not creates_ap:
            S |= x_bit
            
    best_S, best_size = complete_greedily(S)
    
    current_time = time.time()
    
    while current_time < deadline:
        # Start a new annealing run from the best found so far, or a random perturbation of it
        # Perturb best_S: remove k random elements
        k = random.randint(1, 5)
        S = best_S
        # Remove k random elements from S
        bits_in_S = [i for i in range(N) if (S & (1 << i))]
        if len(bits_in_S) > k:
            to_remove = random.sample(bits_in_S, k)
            for idx in to_remove:
                S &= ~(1 << idx)
        else:
            # If S is too small, just start fresh with a random greedy
            S = 0
            order = list(range(N))
            random.shuffle(order)
            for x_idx in order:
                x_bit = 1 << x_idx
                creates_ap = False
                for mask in triples_for[x_idx]:
                    if (S & mask) == mask:
                        creates_ap = True
                        break
                if not creates_ap:
                    S |= x_bit

        # Simulated Annealing
        T = 5.0
        cooling = 0.95
        num_aps = 0 # We don't track num_aps exactly for speed, we just check validity at the end or use a proxy
        # Actually, for SA to work well with "no AP" constraint, we need to know if we are valid.
        # The current code tracks num_aps. Let's stick to that but optimize.
        
        # Recalculate num_aps for current S
        # This is expensive O(N * avg_degree). avg_degree is ~100. N=512. 512*100 = 50k ops.
        # We can do this once per restart.
        num_aps = 0
        for x_idx in range(N):
            if S & (1 << x_idx):
                for mask in triples_for[x_idx]:
                    if (S & mask) == mask:
                        num_aps += 1
        # Note: Each AP is counted 4 times (once for each element).
        # So actual APs = num_aps // 4.
        # But for delta calculation, we just need consistency.
        # Let's use "violations" = num_aps.
        
        size = bin(S).count('1')
        
        # SA Loop
        while T > 0.05 and time.time() - current_time < 1.0: # Limit each SA run to ~1s
            x_idx = random.randint(0, N - 1)
            x_bit = 1 << x_idx
            
            if S & x_bit:
                # Remove x
                # Count APs destroyed
                k_destroyed = 0
                for mask in triples_for[x_idx]:
                    if (S & mask) == mask:
                        k_destroyed += 1
                
                new_num_aps = num_aps - k_destroyed
                new_size = size - 1
                # Objective: Maximize size, Minimize violations.
                # Score = size - M * violations. Let M be large.
                # Or simply: accept if (new_size - new_num_aps) > (size - num_aps)
                # Let's use F = size - 10 * num_aps
                delta_F = (new_size - 10 * new_num_aps) - (size - 10 * num_aps)
                
                if delta_F > 0 or random.random() < math.exp(-delta_F / T):
                    S ^= x_bit
                    num_aps = new_num_aps
                    size = new_size
            else:
                # Add x
                k_created = 0
                for mask in triples_for[x_idx]:
                    if (S & mask) == mask:
                        k_created += 1
                
                new_num_aps = num_aps + k_created
                new_size = size + 1
                delta_F = (new_size - 10 * new_num_aps) - (size - 10 * num_aps)
                
                if delta_F > 0 or random.random() < math.exp(-delta_F / T):
                    S |= x_bit
                    num_aps = new_num_aps
                    size = new_size
            
            T *= cooling

        # Polish: Local Search (Greedy Hill Climbing)
        # Try to add any element that doesn't create an AP
        # And try to remove elements that are "useless" (don't block any additions? No, just try to add)
        # Actually, a better polish is:
        # 1. Try to add any valid element.
        # 2. Try to swap: remove x, add y if it results in larger set?
        
        # Simple Polish: Add all possible valid elements
        S, size = complete_greedily(S)
        
        # Check if this is better
        if size > best_size:
            best_size = size
            best_S = S
            
        current_time = time.time()

    # Convert bit mask to sorted list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
