import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(N):
    """Precompute LCM table for all pairs in [1, N]."""
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, N):
    """
    Precompute conflict bitsets.
    conflict[x][a] is a bitmask of b such that (a,b,x) is a bad triple.
    i.e., lcm(a,b) == lcm(a,x) == lcm(b,x).
    """
    # conflict[x][a] will hold a bitmask of integers b
    # We use a list of lists of integers (bitmasks)
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    # To build this efficiently, we can iterate over pairs (a,b) and find x,
    # or iterate over x and pairs (a,b).
    # Given N=128, O(N^3) is ~2M operations, which is fast enough in Python if optimized.
    # Let's iterate x, then a, then b.
    
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            la_x = lcm_table[a][x]
            mask = 0
            for b in range(1, N + 1):
                if b == x or b == a:
                    continue
                if lcm_table[a][b] == la_x and lcm_table[b][x] == la_x:
                    mask |= (1 << b)
            conflict[x][a] = mask
            
    return conflict

def _popcount(m):
    return bin(m).count('1')

def _mask_to_list(mask):
    return [i for i in range(1, 129) if mask & (1 << i)]

def _list_to_mask(lst):
    m = 0
    for x in lst:
        m |= (1 << x)
    return m

def _greedy_construct(order, conflict, N):
    """
    Greedily add numbers in the given order if they don't form a bad triple.
    Uses precomputed conflict bitmasks.
    """
    selected_mask = 0
    for x in order:
        if selected_mask & (1 << x):
            continue
        # Check if x can be added
        ok = True
        # For each a in selected, check if conflict[x][a] intersects with selected
        # We can iterate through set bits in selected_mask
        m = selected_mask
        while m:
            # Get lowest set bit
            a_bit = m & (-m)
            a = a_bit.bit_length() - 1
            if conflict[x][a] & selected_mask:
                ok = False
                break
            m ^= a_bit
        if ok:
            selected_mask |= (1 << x)
    return selected_mask

def _local_search_improve(mask, conflict, N, deadline):
    """
    Try to improve the mask by swapping elements or adding new ones.
    """
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # 1. Try adding each excluded number
        # We can iterate through all numbers 1..N and check if not in mask
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if mask & (1 << x):
                continue
            # Check if x can be added
            ok = True
            m = mask
            while m:
                a_bit = m & (-m)
                a = a_bit.bit_length() - 1
                if conflict[x][a] & mask:
                    ok = False
                    break
                m ^= a_bit
            if ok:
                mask |= (1 << x)
                improved = True
                break
        
        # 2. Try swapping: remove one element, try to add a different one
        if not improved:
            # Get list of selected elements for iteration
            selected = _mask_to_list(mask)
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                candidate_remove = selected[i]
                new_mask = mask & ~(1 << candidate_remove)
                
                # Try to add any x not in new_mask
                for x in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if new_mask & (1 << x):
                        continue
                    # Check if x can be added to new_mask
                    ok = True
                    m = new_mask
                    while m:
                        a_bit = m & (-m)
                        a = a_bit.bit_length() - 1
                        if conflict[x][a] & new_mask:
                            ok = False
                            break
                        m ^= a_bit
                    if ok:
                        # We found a valid swap that maintains or increases size
                        # Since we removed 1 and added 1, size is same.
                        # But maybe we can add more later.
                        # Actually, simple swap keeps size same.
                        # We want to increase size. 
                        # The "add" phase above handles size increase.
                        # This swap phase is useful if we are stuck at a local maximum 
                        # where we need to rearrange to allow future additions.
                        mask = new_mask | (1 << x)
                        improved = True
                        break
                if improved:
                    break
    
    return mask

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(N)
    
    # Precompute conflict bitsets
    conflict = _build_conflict_table(lcm_table, N)
    
    # Helper to check primality and prime powers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Primes and Prime Powers first
    order1 = primes + prime_powers + others
    mask1 = _greedy_construct(order1, conflict, N)
    size1 = _popcount(mask1)
    if size1 > best_size:
        best_mask = mask1
        best_size = size1
    
    # Strategy 2: Standard 1..128
    mask2 = _greedy_construct(numbers, conflict, N)
    size2 = _popcount(mask2)
    if size2 > best_size:
        best_mask = mask2
        best_size = size2
        
    # Strategy 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    mask3 = _greedy_construct(order3, conflict, N)
    size3 = _popcount(mask3)
    if size3 > best_size:
        best_mask = mask3
        best_size = size3

    # Try many random shuffles
    random.seed(42)
    # We have ~80s left. Greedy is fast. Let's run as many as possible.
    # Local search will take the rest.
    search_deadline = deadline - 10 # Reserve 10s for local search
    
    for i in range(5000):
        if time.time() > search_deadline:
            break
        random.shuffle(numbers)
        res_mask = _greedy_construct(numbers, conflict, N)
        res_size = _popcount(res_mask)
        if res_size > best_size:
            best_mask = res_mask
            best_size = res_size
    
    # Local search improvement on the best result
    if time.time() < deadline - 2:
        best_mask = _local_search_improve(best_mask, conflict, N, deadline - 2)
        # Update best size just in case, though local search shouldn't decrease it
        final_size = _popcount(best_mask)
        if final_size > best_size:
            best_size = final_size

    return _mask_to_list(best_mask)

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    def _lcm_check(a, b):
        return a * b // gcd(a, b)
    
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            for k in range(j+1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if _lcm_check(a,b) == _lcm_check(a,c) == _lcm_check(b,c):
                    valid = False
                    print(f"Invalid triple: {a},{b},{c}")
    print(f"Valid: {valid}")
