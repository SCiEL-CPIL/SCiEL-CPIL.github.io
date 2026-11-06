"""Combined program: bitset compatibility + divisor seeds + effective local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 512
    
    # Precompute compatibility bitsets
    # compat[i] is a bitset where bit (j-1) is set iff lcm(i,j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 1 << (i - 1)  # always compatible with self
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            if i // g * j <= N:
                mask |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
        compat[i] = mask
    
    def is_compatible(x, S_mask):
        """Check if x is compatible with all elements in S_mask."""
        return (compat[x] & S_mask) == S_mask
    
    def greedy_fill(order, S_mask, S_list):
        """Greedily add elements from order that are compatible with current set."""
        for x in order:
            if not (S_mask & (1 << (x - 1))):
                if is_compatible(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    S_list.append(x)
        return S_mask, S_list
    
    def local_search(S_mask, S_list):
        """Try to increase set size by adding free elements and performing swaps."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Add any element that has no conflicts
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    break
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_list.append(x)
                        improved = True
            
            # Phase 2: Try 1-for-1 swaps that might enable future additions
            # Remove one element, see if we can add two or more
            for idx in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                removed = S_list[idx]
                S_mask_tmp = S_mask & ~(1 << (removed - 1))
                
                # Count how many new elements we could add
                additions = 0
                for x in range(1, N + 1):
                    if not (S_mask_tmp & (1 << (x - 1))):
                        if is_compatible(x, S_mask_tmp):
                            additions += 1
                
                if additions >= 2:  # Net gain of at least 1
                    # Perform the swap: remove 'removed', add all compatible new elements
                    S_mask = S_mask_tmp
                    S_list = [x for x in S_list if x != removed]
                    for x in range(1, N + 1):
                        if not (S_mask & (1 << (x - 1))):
                            if is_compatible(x, S_mask):
                                S_mask |= (1 << (x - 1))
                                S_list.append(x)
                    improved = True
                    break  # Restart the loop with the new set
        
        return S_mask, S_list
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs
    
    best_size = 0
    best_set = []
    
    # Strategy 1: Divisor-based seeds (strong starting points)
    # Try divisors of highly composite numbers and other M values
    candidate_Ms = list(range(1, N + 1))
    # Prioritize highly composite numbers
    priority_Ms = [504, 480, 420, 360, 336, 320, 288, 240, 210, 180, 
                   168, 144, 120, 96, 84, 72, 60, 48, 42, 36, 30, 24, 20, 18, 16, 12, 10, 8, 6, 4, 3, 2, 1]
    
    for M in priority_Ms:
        if time.time() > deadline - 3:
            break
        divs = get_divisors(M)
        divs.sort()
        S_mask = 0
        S_list = []
        for d in divs:
            S_mask |= (1 << (d - 1))
        # Greedy fill with remaining numbers
        remaining = [x for x in range(1, N + 1) if x not in set(divs)]
        S_mask, S_list = greedy_fill(remaining, S_mask, S_list)
        S_mask, S_list = local_search(S_mask, S_list)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
    
    # Strategy 2: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -len([d for d in range(1, x+1) if x % d == 0])),
        sorted(range(1, N + 1), key=lambda x: x),
    ]
    
    for order in orderings:
        if time.time() > deadline - 3:
            break
        S_mask = 0
        S_list = []
        S_mask, S_list = greedy_fill(order, S_mask, S_list)
        S_mask, S_list = local_search(S_mask, S_list)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
    
    # Strategy 3: Random restarts
    random.seed(42)
    numbers = list(range(1, N + 1))
    
    for restart in range(2000):
        if time.time() > deadline - 2:
            break
        
        random.shuffle(numbers)
        S_mask = 0
        S_list = []
        S_mask, S_list = greedy_fill(numbers, S_mask, S_list)
        S_mask, S_list = local_search(S_mask, S_list)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
    
    # Strategy 4: Targeted local search on best set with more aggressive swaps
    if best_set and time.time() < deadline - 2:
        S_mask = 0
        for x in best_set:
            S_mask |= (1 << (x - 1))
        S_list = best_set[:]
        
        # More thorough local search: try removing pairs of elements
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 2:
                        break
                    r1, r2 = S_list[i], S_list[j]
                    S_mask_tmp = S_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                    
                    additions = 0
                    add_list = []
                    for x in range(1, N + 1):
                        if not (S_mask_tmp & (1 << (x - 1))):
                            if is_compatible(x, S_mask_tmp):
                                additions += 1
                                add_list.append(x)
                    
                    if additions >= 3:  # Net gain of at least 1
                        S_mask = S_mask_tmp
                        S_list = [x for x in S_list if x not in (r1, r2)]
                        for x in add_list:
                            S_mask |= (1 << (x - 1))
                            S_list.append(x)
                        improved = True
                        break
                if improved:
                    break
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
    
    if not best_set:
        best_set = [1]
    
    return best_set

# EVOLVE-BLOCK-END
