"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    # Bitmask representation: bit i is set if value i is in the set
    # We use values 1..512, so bits 1..512 (bit 0 unused)
    
    def mask_to_list(mask):
        """Convert bitmask to sorted list of values."""
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def list_to_mask(lst):
        """Convert list of values to bitmask."""
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask
    
    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        valid_diffs = set()
        # For each possible difference d (1 to 255, since a+2d <= 512 means d <= 255)
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are all in the set
            # a must be in 1..512, a+d in 1..512, a+2d in 1..512
            # So a is in 1..(512-2d)
            found = False
            # We can check by iterating a, but a faster way:
            # For a fixed d, we need three consecutive elements in the arithmetic progression with step d
            # Let's iterate a from 1 to 512-2d
            # But we can optimize: check if (mask & (mask >> d) & (mask >> 2*d)) has any bit set
            # (mask >> d) has bit i set if mask has bit i+d set
            # (mask >> 2*d) has bit i set if mask has bit i+2d set
            # So (mask & (mask >> d) & (mask >> 2*d)) has bit a set if a, a+d, a+2d are all in mask
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                valid_diffs.add(d)
        return len(valid_diffs)
    
    def score_list(lst):
        """Score a list of 32 distinct integers."""
        mask = list_to_mask(lst)
        return score_mask(mask)
    
    def random_initial():
        """Generate a random initial set of 32 distinct integers in 1..512."""
        return random.sample(range(1, MAX_VAL + 1), N)
    
    def local_search(lst, max_iters=2000):
        """
        Local search: try swapping elements or moving elements to improve score.
        Returns improved list and score.
        """
        current_score = score_list(lst)
        best_lst = lst[:]
        best_score = current_score
        current_lst = lst[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                i, j = random.sample(range(N), 2)
                # Swap current_lst[i] and current_lst[j]
                current_lst[i], current_lst[j] = current_lst[j], current_lst[i]
                new_score = score_list(current_lst)
                
                if new_score > best_score:
                    best_score = new_score
                    best_lst = current_lst[:]
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    current_lst[i], current_lst[j] = current_lst[j], current_lst[i]
            
            # Try random moves: replace one element with a random new one
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                i = random.randint(0, N - 1)
                old_val = current_lst[i]
                new_val = random.randint(1, MAX_VAL)
                
                if new_val == old_val:
                    continue
                if new_val in current_lst:
                    continue
                
                current_lst[i] = new_val
                new_score = score_list(current_lst)
                
                if new_score > best_score:
                    best_score = new_score
                    best_lst = current_lst[:]
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    current_lst[i] = old_val
            
            if not improved:
                # No improvement in this iteration, try a few more random restarts within local search
                # or break out
                pass
        
        return best_lst, best_score
    
    # Multi-start local search
    best_overall_lst = None
    best_overall_score = 0
    
    # Start with the seed solution
    seed_lst = list(range(1, 33))
    seed_score = score_list(seed_lst)
    best_overall_score = seed_score
    best_overall_lst = seed_lst[:]
    
    num_starts = 0
    while time.time() < deadline - 10.0:
        num_starts += 1
        
        # Generate a diverse initial solution
        if num_starts % 3 == 1:
            # Random
            init_lst = random_initial()
        elif num_starts % 3 == 2:
            # Structured: arithmetic progression-like
            base = random.randint(1, 50)
            step = random.randint(5, 15)
            init_lst = []
            for i in range(N):
                val = base + i * step
                if val <= MAX_VAL:
                    init_lst.append(val)
            # Fill remaining with random
            while len(init_lst) < N:
                val = random.randint(1, MAX_VAL)
                if val not in init_lst:
                    init_lst.append(val)
            init_lst = init_lst[:N]
        else:
            # Structured: two interleaved progressions
            base1 = random.randint(1, 30)
            base2 = random.randint(30, 60)
            step1 = random.randint(8, 16)
            step2 = random.randint(8, 16)
            init_lst = []
            for i in range(N // 2):
                v1 = base1 + i * step1
                v2 = base2 + i * step2
                if v1 <= MAX_VAL and v1 not in init_lst:
                    init_lst.append(v1)
                if v2 <= MAX_VAL and v2 not in init_lst:
                    init_lst.append(v2)
            while len(init_lst) < N:
                val = random.randint(1, MAX_VAL)
                if val not in init_lst:
                    init_lst.append(val)
            init_lst = init_lst[:N]
        
        # Run local search from this initial solution
        improved_lst, improved_score = local_search(init_lst)
        
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_lst = improved_lst[:]
    
    # Final check: make sure we return exactly 32 distinct integers in 1..512
    result = best_overall_lst[:]
    # Ensure uniqueness and range
    seen = set()
    clean_result = []
    for v in result:
        if 1 <= v <= MAX_VAL and v not in seen:
            seen.add(v)
            clean_result.append(v)
    
    # If we don't have 32, fill with random
    while len(clean_result) < N:
        val = random.randint(1, MAX_VAL)
        if val not in seen:
            seen.add(val)
            clean_result.append(val)
    
    clean_result = clean_result[:N]
    
    return sorted(clean_result)

# EVOLVE-BLOCK-END
