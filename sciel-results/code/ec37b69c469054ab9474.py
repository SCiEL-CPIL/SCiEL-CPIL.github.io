"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import gcd

def construct():
    deadline = time.time() + 82  # 82s budget, 8s safety margin
    N = 512
    LIMIT = 512
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute compatibility bitsets: compat[i] has bit j set if lcm(i,j) <= 512
    # This is the key speedup - O(1) compatibility check via bitwise AND
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        bits = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= LIMIT:
                bits |= (1 << j)
        compat[i] = bits
    
    # Full mask for elements 1..N
    full_mask = 0
    for j in range(1, N + 1):
        full_mask |= (1 << j)
    
    def is_compatible_with_all(elem, current_bits):
        """Check if elem is compatible with all elements in current_bits (excluding itself)."""
        return (compat[elem] & current_bits) == current_bits
    
    def greedy_expand(initial_set, deadline_time=None):
        """Greedy expansion: try to add elements that maintain validity."""
        current = set(initial_set)
        current_bits = 0
        for x in current:
            current_bits |= (1 << x)
        
        # Try all elements from 1 to 512, adding if valid
        for candidate in range(1, N + 1):
            if deadline_time is not None and time.time() > deadline_time:
                break
            if candidate not in current and is_compatible_with_all(candidate, current_bits):
                current.add(candidate)
                current_bits |= (1 << candidate)
        return current
    
    def local_search_swap(current, deadline_time=None, max_passes=15):
        """Local search with swap moves: remove one element, add another."""
        current = set(current)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if deadline_time is not None and time.time() > deadline_time:
                break
            improved = False
            passes += 1
            current_bits = 0
            for x in current:
                current_bits |= (1 << x)
            
            # Try swapping each element with a non-current element
            for x in list(current):
                if deadline_time is not None and time.time() > deadline_time:
                    break
                # Remove x
                without_x = current_bits & ~(1 << x)
                # Find best replacement: an element not in current that is compatible with all in without_x
                best_candidate = None
                best_score = -1
                for y in range(1, N + 1):
                    if y in current:
                        continue
                    if not is_compatible_with_all(y, without_x):
                        continue
                    # Score: number of elements y is compatible with (more flexible is better)
                    score = bin(compat[y] & full_mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_candidate = y
                if best_candidate is not None:
                    current.remove(x)
                    current.add(best_candidate)
                    improved = True
        return current
    
    def local_search_remove_add(current, deadline_time=None, max_passes=10):
        """Try removing an element and then greedily adding back what fits."""
        current = set(current)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if deadline_time is not None and time.time() > deadline_time:
                break
            improved = False
            passes += 1
            
            for x in list(current):
                if deadline_time is not None and time.time() > deadline_time:
                    break
                # Try removing x
                trial = current - {x}
                trial_bits = 0
                for y in trial:
                    trial_bits |= (1 << y)
                
                # Greedily add back what fits
                expanded = set(trial)
                eb = trial_bits
                for candidate in range(1, N + 1):
                    if candidate not in expanded and is_compatible_with_all(candidate, eb):
                        expanded.add(candidate)
                        eb |= (1 << candidate)
                
                if len(expanded) > len(current):
                    current = expanded
                    improved = True
        return current
    
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    # Generate seed sets: divisors of highly composite numbers
    hcn_candidates = list(range(1, N + 1))
    hcn_candidates.sort(key=lambda x: -count_divisors(x))
    
    best_set = set()
    best_size = 0
    
    for m in hcn_candidates:
        if time.time() > deadline:
            break
        
        initial = get_divisors(m)
        initial = {x for x in initial if 1 <= x <= N}
        
        if not initial:
            continue
        
        # Verify initial set is valid using bitsets
        valid = True
        initial_bits = 0
        for x in initial:
            initial_bits |= (1 << x)
        for x in initial:
            if not is_compatible_with_all(x, initial_bits & ~(1 << x)):
                valid = False
                break
        
        if not valid:
            continue
        
        # Greedy expansion
        expanded = greedy_expand(initial, deadline_time=deadline)
        
        # Local search with swaps
        if time.time() < deadline - 5:
            expanded = local_search_swap(expanded, deadline_time=deadline, max_passes=8)
        
        # Local search with remove-add
        if time.time() < deadline - 3:
            expanded = local_search_remove_add(expanded, deadline_time=deadline, max_passes=5)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = set(expanded)
    
    # Also try starting from small sets of prime powers
    prime_powers = []
    for p in [2, 3, 5, 7, 11, 13]:
        pp = p
        while pp <= N:
            prime_powers.append(pp)
            pp *= p
    
    # Try combinations of 2-3 prime powers as seeds
    for i in range(len(prime_powers)):
        if time.time() > deadline:
            break
        for j in range(i + 1, min(i + 10, len(prime_powers))):
            if time.time() > deadline:
                break
            seed = {prime_powers[i], prime_powers[j]}
            # Check if seed is valid
            if lcm(prime_powers[i], prime_powers[j]) > N:
                continue
            # Expand
            expanded = greedy_expand(seed, deadline_time=deadline)
            if time.time() < deadline - 5:
                expanded = local_search_swap(expanded, deadline_time=deadline, max_passes=5)
            if len(expanded) > best_size:
                best_size = len(expanded)
                best_set = set(expanded)
    
    # Fallback: at least return {1}
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
