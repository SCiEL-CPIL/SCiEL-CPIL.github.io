import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Bitset-based verification for speed
    # bits is an integer where bit i (0-indexed) represents whether element i+1 is in the set
    
    def bits_to_set(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]
    
    def set_to_bits(s):
        bits = 0
        for x in s:
            bits |= (1 << (x - 1))
        return bits
    
    def can_add(bits, x):
        """Check if element x (1-indexed) can be added to the set represented by bits."""
        bit = (1 << (x - 1))
        if bits & bit:
            return False
        
        # Check 1: x, x+d, x+2d, x+3d (x is first term)
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1 and (bits >> (x + 3*d - 1)) & 1:
                return False
        
        # Check 2: x-3d, x-2d, x-d, x (x is last term)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x - 3*d - 1)) & 1 and (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
                return False
        
        # Check 3: x-2d, x-d, x, x+d (x is third term)
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
                return False
        
        # Check 4: x-d, x, x+d, x+2d (x is second term)
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
                return False
        
        return True
    
    def is_4ap_free_bits(bits, x):
        """Check if element x (1-indexed) is 4AP-free in the set represented by bits (assuming x is already in the set)."""
        bit = (1 << (x - 1))
        if not (bits & bit):
            return True
        
        # Check 1: x, x+d, x+2d, x+3d (x is first term)
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1 and (bits >> (x + 3*d - 1)) & 1:
                return False
        
        # Check 2: x-3d, x-2d, x-d, x (x is last term)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x - 3*d - 1)) & 1 and (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
                return False
        
        # Check 3: x-2d, x-d, x, x+d (x is third term)
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
                return False
        
        # Check 4: x-d, x, x+d, x+2d (x is second term)
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
                return False
        
        return True
    
    def greedy_forward(bits):
        """Greedy: add elements in order 1, 2, ..., N if possible."""
        for x in range(1, N + 1):
            if can_add(bits, x):
                bits |= (1 << (x - 1))
        return bits
    
    def greedy_reverse(bits):
        """Greedy: add elements in order N, N-1, ..., 1 if possible."""
        for x in range(N, 0, -1):
            if can_add(bits, x):
                bits |= (1 << (x - 1))
        return bits
    
    def greedy_random_order(bits):
        """Greedy: add elements in random order if possible."""
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(bits, x):
                bits |= (1 << (x - 1))
        return bits
    
    def greedy_density_weighted(bits):
        """Greedy: prioritize positions with fewer conflicts."""
        candidates = list(range(1, N + 1))
        
        # Compute conflict count for each candidate
        def conflict_count(x):
            count = 0
            # Check 1: x, x+d, x+2d, x+3d
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1 and (bits >> (x + 3*d - 1)) & 1:
                    count += 1
            # Check 2: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (bits >> (x - 3*d - 1)) & 1 and (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
                    count += 1
            # Check 3: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
                    count += 1
            # Check 4: x-d, x, x+d, x+2d
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
                    count += 1
            return count
        
        # Sort by conflict count (fewer conflicts first), with random tie-breaking
        candidates.sort(key=lambda x: (conflict_count(x), random.random()))
        
        for x in candidates:
            if can_add(bits, x):
                bits |= (1 << (x - 1))
        return bits
    
    def cantor_set_bits():
        """Generate Cantor set-like seed: numbers with only 0s and 2s in base 3."""
        bits = 0
        for i in range(N):
            x = i + 1
            # Check if x has only 0s and 2s in base 3
            valid = True
            temp = x
            while temp > 0:
                digit = temp % 3
                if digit == 1:
                    valid = False
                    break
                temp //= 3
            if valid:
                bits |= (1 << i)
        return bits
    
    def local_improve(bits, time_limit):
        """Local search with add and swap operations."""
        current_size = bin(bits).count('1')
        best_size = current_size
        best_bits = bits
        
        while time.time() < time_limit:
            improved = False
            
            # Try Add: sample random candidates
            for _ in range(50):
                x = random.randint(1, N)
                if not (bits >> (x - 1)) & 1:
                    if can_add(bits, x):
                        bits |= (1 << (x - 1))
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_bits = bits
                        improved = True
                        break
            
            # Try Swap: pick random y in S, random x not in S, try swapping
            if current_size > 0:
                for _ in range(50):
                    y = random.randint(1, N)
                    if (bits >> (y - 1)) & 1:
                        x = random.randint(1, N)
                        if not (bits >> (x - 1)) & 1:
                            # Try removing y and adding x
                            new_bits = bits & ~(1 << (y - 1))
                            if can_add(new_bits, x):
                                bits = new_bits | (1 << (x - 1))
                                if current_size > best_size:
                                    best_size = current_size
                                    best_bits = bits
                                improved = True
                                break
                            else:
                                # Revert
                                bits |= (1 << (y - 1))
            
            if not improved:
                # No improvement in this iteration, maybe break early or continue
                pass
        
        return best_bits, best_size
    
    def pair_swap_improve(bits, time_limit):
        """Local search with pair swaps: remove two elements, add two different ones."""
        current_size = bin(bits).count('1')
        best_size = current_size
        best_bits = bits
        
        while time.time() < time_limit:
            improved = False
            
            # Try pair swap: remove y1, y2 from S, add x1, x2 not in S
            for _ in range(30):
                y1 = random.randint(1, N)
                y2 = random.randint(1, N)
                if (bits >> (y1 - 1)) & 1 and (bits >> (y2 - 1)) & 1 and y1 != y2:
                    x1 = random.randint(1, N)
                    x2 = random.randint(1, N)
                    if not (bits >> (x1 - 1)) & 1 and not (bits >> (x2 - 1)) & 1 and x1 != x2:
                        # Try removing y1, y2 and adding x1, x2
                        new_bits = bits & ~(1 << (y1 - 1)) & ~(1 << (y2 - 1))
                        if can_add(new_bits, x1) and can_add(new_bits | (1 << (x1 - 1)), x2):
                            bits = new_bits | (1 << (x1 - 1)) | (1 << (x2 - 1))
                            if current_size > best_size:
                                best_size = current_size
                                best_bits = bits
                            improved = True
                            break
                        else:
                            # Revert
                            bits |= (1 << (y1 - 1)) | (1 << (y2 - 1))
            
            if not improved:
                pass
        
        return best_bits, best_size
    
    # Initialize with multiple strategies
    best_bits = 0
    best_size = 0
    
    # Strategy 1: Cantor set seed
    cantor_bits = cantor_set_bits()
    cantor_size = bin(cantor_bits).count('1')
    if cantor_size > best_size:
        best_size = cantor_size
        best_bits = cantor_bits
    
    # Strategy 2: Forward greedy
    fwd_bits = greedy_forward(0)
    fwd_size = bin(fwd_bits).count('1')
    if fwd_size > best_size:
        best_size = fwd_size
        best_bits = fwd_bits
    
    # Strategy 3: Reverse greedy
    rev_bits = greedy_reverse(0)
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_size = rev_size
        best_bits = rev_bits
    
    # Strategy 4: Random order greedy (multiple restarts)
    for _ in range(5):
        rnd_bits = greedy_random_order(0)
        rnd_size = bin(rnd_bits).count('1')
        if rnd_size > best_size:
            best_size = rnd_size
            best_bits = rnd_bits
    
    # Strategy 5: Density-weighted greedy
    dw_bits = greedy_density_weighted(0)
    dw_size = bin(dw_bits).count('1')
    if dw_size > best_size:
        best_size = dw_size
        best_bits = dw_bits
    
    # Local improve the best found so far
    t_start = time.time()
    improved_bits, improved_size = local_improve(best_bits, t_start + 10)
    if improved_size > best_size:
        best_size = improved_size
        best_bits = improved_bits
    
    # Pair swap improve
    t_start = time.time()
    improved_bits, improved_size = pair_swap_improve(best_bits, t_start + 10)
    if improved_size > best_size:
        best_size = improved_size
        best_bits = improved_bits
    
    # Main loop: continue improving with local search
    current_bits = best_bits
    current_size = best_size
    
    while time.time() < deadline - 5:
        t_start = time.time()
        
        # Alternate between local improve and pair swap improve
        if random.random() < 0.5:
            new_bits, new_size = local_improve(current_bits, t_start + 5)
        else:
            new_bits, new_size = pair_swap_improve(current_bits, t_start + 5)
        
        if new_size > best_size:
            best_size = new_size
            best_bits = new_bits
            current_bits = best_bits
        else:
            # If no improvement, try a fresh random start occasionally
            if time.time() - start_time > 20 and random.random() < 0.1:
                fresh_bits = greedy_random_order(0)
                fresh_size = bin(fresh_bits).count('1')
                if fresh_size > best_size:
                    best_size = fresh_size
                    best_bits = fresh_bits
                    current_bits = best_bits
                else:
                    # Try improving the fresh start
                    fresh_improved_bits, fresh_improved_size = local_improve(fresh_bits, time.time() + 3)
                    if fresh_improved_size > best_size:
                        best_size = fresh_improved_size
                        best_bits = fresh_improved_bits
                        current_bits = best_bits
    
    # Convert best_bits to sorted list
    result = bits_to_set(best_bits)
    return result
