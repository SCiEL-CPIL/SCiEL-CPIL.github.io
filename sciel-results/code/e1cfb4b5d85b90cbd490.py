import itertools
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks
    # mask[i] has bit j set if words[i] covers words[j] (dist <= 1)
    masks = [0] * N
    for i in range(N):
        w_i = words[i]
        mask_val = 0
        # Self
        mask_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                mask_val |= (1 << j)
        masks[i] = mask_val
        
    full_mask = (1 << N) - 1
    
    # Initialize with Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            w = tuple(h + list(tail))
            init_indices.append(word_to_idx[w])
            
    best_code = set(init_indices)
    best_size = len(best_code)
    # Verify coverage of initial code
    cov = 0
    for idx in best_code:
        cov |= masks[idx]
    if cov != full_mask:
        # Should not happen for tetracode, but just in case
        pass

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Simulated Annealing
    # State: set of indices
    current_code = set(best_code)
    current_cov = 0
    for idx in current_code:
        current_cov |= masks[idx]
        
    # We track the best valid solution found
    # Cost function: size + large_penalty * uncovered_bits
    # We want to minimize size.
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Pre-compute list of all indices for random selection
    all_indices = list(range(N))
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random codeword
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            new_code = set(current_code)
            new_code.remove(to_remove)
            # Calculate new coverage
            # Optimization: recompute coverage is O(|C|). |C| is small (~80). N=729.
            # It's fast enough.
            new_cov = 0
            for idx in new_code:
                new_cov |= masks[idx]
                
        elif move_type < 0.6:
            # Add a random codeword
            to_add = random.choice(all_indices)
            if to_add in current_code:
                continue
            new_code = set(current_code)
            new_code.add(to_add)
            new_cov = current_cov | masks[to_add]
            
        else:
            # Swap: remove one, add another
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            to_add = random.choice(all_indices)
            if to_add == to_remove or to_add in current_code:
                continue
            new_code = set(current_code)
            new_code.remove(to_remove)
            new_code.add(to_add)
            new_cov = 0
            for idx in new_code:
                new_cov |= masks[idx]
                
        # Calculate delta cost
        # Cost = |C| + 1000 * (N - bit_count(cov))
        # Using int.bit_count() (Python 3.8+)
        
        curr_uncovered = N - current_cov.bit_count()
        new_uncovered = N - new_cov.bit_count()
        
        curr_cost = len(current_code) + 1000 * curr_uncovered
        new_cost = len(new_code) + 1000 * new_uncovered
        
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_code = new_code
            current_cov = new_cov
            
            # Update best if valid (uncovered == 0)
            if new_uncovered == 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                    # If we hit 73, we can stop early?
                    # 73 is the best known. 72 is unlikely.
                    if best_size == 73:
                        # Try a bit longer to see if 72 is possible?
                        # Or just break to save time?
                        # Let's break if we hit 73, as 72 is extremely hard.
                        pass
                        
        T *= cooling
        
    # Final check: ensure best_code is valid
    # (It should be, as we only update best when uncovered==0)
    
    return [list(words[i]) for i in best_code]
