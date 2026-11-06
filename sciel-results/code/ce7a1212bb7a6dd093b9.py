import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_points = 16
    all_points = list(range(n_points))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples_list):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute conflict bits
    # conflict_bits[i] is a bitmask of indices j such that triple i and j share >= 2 points
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(i + 1, num_triples):
            mj = triple_masks[j]
            inter = mi & mj
            # Check if inter has more than 1 bit set
            if inter & (inter - 1):
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)

    # Helper to calculate independence number alpha
    # active_mask: bitmask of active triple indices
    def calculate_alpha(active_mask):
        if active_mask == 0:
            return 16
            
        # is_covered[s] is True if subset s contains at least one active triple
        # Size is 2^16
        is_covered = [False] * (1 << n_points)
        
        # Iterate through active triples
        # Extract indices from active_mask
        # active_mask is a Python int, so we can iterate bits
        m = active_mask
        while m:
            # Get lowest set bit index
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            tm = triple_masks[idx]
            # Supersets of tm are tm | x where x is subset of ~tm
            # ~tm in 16 bits is (1 << 16) - 1 - tm
            comp = ((1 << n_points) - 1) ^ tm
            # Iterate all subsets of comp
            # There are 2^(16-3) = 8192 subsets
            # We can iterate submasks of comp
            sub = comp
            while True:
                is_covered[tm | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        
        # Find largest k such that there is an independent set of size k
        # Iterate k from 16 down to 1
        for k in range(n_points, 0, -1):
            # Check all subsets of size k
            # We can just iterate all subsets and check popcount? 
            # No, that's 2^16 = 65536 checks.
            # Better to iterate combinations? 
            # itertools.combinations is slow in Python loops.
            # Let's iterate all masks 0 to 2^16-1 and check popcount == k and not covered.
            # But we can break early if we find one.
            
            # Optimization: Precompute popcount? Or just use bin(x).count('1')?
            # 65536 iterations is fast.
            found = False
            for s in range(1 << n_points):
                if not is_covered[s]:
                    # Check popcount
                    # Using bin(s).count('1') is a bit slow.
                    # We can precompute popcounts for all 2^16 values once outside.
                    pass
                    
            # Let's use a precomputed popcount array for speed
            # We'll create it outside and pass it or use a global
            # For now, let's assume we use a precomputed list `popcounts`
            
        # This implementation inside the function is getting complex.
        # Let's precompute popcounts globally.
        pass

    # Precompute popcounts for all 2^16 masks
    popcounts = [0] * (1 << n_points)
    for i in range(1, 1 << n_points):
        popcounts[i] = popcounts[i >> 1] + (i & 1)

    def calculate_alpha_fast(active_mask):
        if active_mask == 0:
            return 16
            
        is_covered = [False] * (1 << n_points)
        
        m = active_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            tm = triple_masks[idx]
            comp = ((1 << n_points) - 1) ^ tm
            sub = comp
            while True:
                is_covered[tm | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        
        # Now find max k
        # We can iterate k from 16 down to 1
        # For each k, check if any s with popcount k is not covered
        # To speed up, we can just iterate all s, and track max popcount of uncovered s
        max_alpha = 0
        for s in range(1 << n_points):
            if not is_covered[s]:
                pc = popcounts[s]
                if pc > max_alpha:
                    max_alpha = pc
                    if max_alpha == n_points:
                        return n_points
        return max_alpha

    # Generate initial valid packing
    def generate_initial():
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        return active_mask

    current_mask = generate_initial()
    current_alpha = calculate_alpha_fast(current_mask)
    
    best_mask = current_mask
    best_alpha = current_alpha
    
    # Simulated Annealing
    temperature = 1.0
    cooling = 0.999
    # We want to minimize alpha. 
    # Move: Remove one triple, add one triple.
    
    # Precompute list of indices for random selection
    all_indices = list(range(num_triples))
    
    while time.time() < deadline:
        # Pick a random triple to remove
        # We need a list of active indices
        # Extracting list from bitmask every time is O(N)
        active_indices = []
        m = current_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            m ^= lsb
            active_indices.append(idx)
            
        if not active_indices:
            break
            
        remove_idx = random.choice(active_indices)
        
        # New mask without remove_idx
        new_mask = current_mask & ~(1 << remove_idx)
        
        # Pick a random triple to add
        # It must not conflict with new_mask
        # Try a few random candidates
        added = False
        for _ in range(10): # Try 10 random candidates
            add_idx = random.choice(all_indices)
            if (1 << add_idx) & new_mask:
                continue # Already in mask
            if (conflict_bits[add_idx] & new_mask) == 0:
                new_mask |= (1 << add_idx)
                added = True
                break
        
        if not added:
            # If we couldn't add, just accept removal? Or skip move?
            # If we can't add, the system becomes sparser, alpha likely increases.
            # Let's try to add a random triple that was conflicting with remove_idx?
            # Or just skip this move.
            continue
            
        # Calculate new alpha
        new_alpha = calculate_alpha_fast(new_mask)
        
        # Acceptance
        delta_alpha = new_alpha - current_alpha
        
        if delta_alpha < 0:
            # Improvement
            current_mask = new_mask
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_mask = current_mask
        elif delta_alpha == 0:
            # Neutral
            # Accept if we increased density (number of triples)
            # Or with some probability
            if random.random() < 0.5:
                current_mask = new_mask
                current_alpha = new_alpha
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            # Delta is in units of alpha. 1 unit is significant.
            prob = pow(2.71828, -delta_alpha / temperature)
            if random.random() < prob:
                current_mask = new_mask
                current_alpha = new_alpha
                
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    # Return best found
    best_indices = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        m ^= lsb
        best_indices.append(idx)
        
    return [list(all_triples_list[i]) for i in best_indices]

# Run once to check syntax/logic
if __name__ == "__main__":
    res = construct()
    print(len(res))
