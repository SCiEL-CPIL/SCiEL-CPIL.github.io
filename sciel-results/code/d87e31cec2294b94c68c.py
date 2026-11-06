import time
import random
import sys

def construct():
    n = 512
    deadline = time.time() + 55  # Safety margin for 90s limit
    
    # Precompute conflict masks
    # conflicts[x] is a list of bitmasks. If (S & mask) == mask, x cannot be added.
    # x is 1-indexed in problem, 0-indexed in bit operations.
    # Bit i corresponds to element i+1.
    
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
    
    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True
    
    def bit_count(S):
        """Count bits in S."""
        count = 0
        while S:
            S &= S - 1
            count += 1
        return count
    
    def bitset_to_list(S):
        """Convert bitset to list of 1-based indices."""
        result = []
        for i in range(n):
            if S & (1 << i):
                result.append(i + 1)
        return result
    
    def greedy_fill(S):
        """Fill S with all possible elements greedily. Returns new S."""
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(n):
                if not (S & (1 << i)):
                    if can_add(i, S):
                        S |= (1 << i)
                        improved = True
        return S
    
    # Base-3 no-zero-digit construction (digits 1, 2 in base 3)
    def base3_no_zero(x):
        if x == 0:
            return True
        while x > 0:
            if x % 3 == 0:
                return False
            x //= 3
        return True
    
    # Initial set: base-3 no-zero digits
    S_init = 0
    for x in range(1, n + 1):
        if base3_no_zero(x):
            S_init |= (1 << (x - 1))
    
    # Greedy fill from base-3 set
    S_best = greedy_fill(S_init)
    best_size = bit_count(S_best)
    best_list = bitset_to_list(S_best)
    
    # Local search with random kicks
    random.seed(42)
    
    # Also try a pure random greedy start
    S_rand = 0
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if can_add(i, S_rand):
            S_rand |= (1 << i)
    S_rand = greedy_fill(S_rand)
    if bit_count(S_rand) > best_size:
        best_size = bit_count(S_rand)
        S_best = S_rand
        best_list = bitset_to_list(S_rand)
    
    # Local search iterations
    for iteration in range(300):
        if time.time() > deadline - 10:
            break
        
        # Random kick: remove k random elements from best set
        k = random.randint(1, 8)
        S = S_best
        if bit_count(S) > k:
            # Get list of set bits
            set_bits = []
            temp = S
            for i in range(n):
                if temp & (1 << i):
                    set_bits.append(i)
                # temp >>= 1  # Not needed since we check each bit
            
            if len(set_bits) > k:
                to_remove = random.sample(set_bits, k)
                for x in to_remove:
                    S &= ~(1 << x)
        
        # Greedy fill
        S = greedy_fill(S)
        
        if bit_count(S) > best_size:
            best_size = bit_count(S)
            S_best = S
            best_list = bitset_to_list(S)
    
    # Final greedy pass
    S_final = greedy_fill(S_best)
    if bit_count(S_final) > best_size:
        best_size = bit_count(S_final)
        best_list = bitset_to_list(S_final)
    
    return sorted(best_list)
