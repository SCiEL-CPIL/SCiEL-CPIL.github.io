import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute "conflict pairs" for fast validity check?
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    # To check if x can be added to S, we need to check all pairs (a,b) in S.
    # We can precompute for each pair (a,b), the value L = lcm(a,b).
    # Then for candidate x, we check if lcm(a,x) == L and lcm(b,x) == L.
    
    # Helper: Check if adding x to mask S is valid
    # S is a bitmask. We need to iterate over set bits in S.
    # For efficiency, we'll keep a list of selected elements in sync with the mask,
    # or just iterate over the mask. Iterating over 128 bits is fast.
    
    def popcount(m):
        return bin(m).count('1')

    def get_selected(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]

    def can_add(x, mask):
        """Check if x can be added to the set represented by mask."""
        # Get all a in mask
        # For each pair (a,b) in mask, check if {a,b,x} is bad
        # Optimization: Iterate a, then b > a
        # We can extract list of elements once
        selected = get_selected(mask)
        n_sel = len(selected)
        for i in range(n_sel):
            a = selected[i]
            la_x = lcm_table[a][x]
            # If la_x is not the same as lcm(a,b), it can't be a bad triple with b?
            # Bad triple condition: lcm(a,b) == lcm(a,x) == lcm(b,x)
            # So for a fixed a, we need lcm(a,b) == la_x.
            # We can filter b's? No, just check.
            for j in range(i + 1, n_sel):
                b = selected[j]
                if lcm_table[a][b] == la_x:
                    if lcm_table[b][x] == la_x:
                        return False
        return True

    def is_valid_set(mask):
        """Verify the whole set is valid (for final check or debugging)."""
        selected = get_selected(mask)
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                la_b = lcm_table[a][b]
                for k in range(j + 1, len(selected)):
                    c = selected[k]
                    if lcm_table[a][c] == la_b and lcm_table[b][c] == la_b:
                        return False
        return True

    # Initial Greedy Construction
    def greedy_construct(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << x)
        return mask

    # Generate several initial solutions
    best_mask = 0
    best_size = 0
    
    # Order 1: Natural
    m1 = greedy_construct(numbers)
    best_mask = m1
    best_size = popcount(m1)
    
    # Order 2: Primes/Prime Powers first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
            
    def is_prime_power(n):
        if n < 2: return False
        p = n
        while p > 1:
            if p * p > n: break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    m2 = greedy_construct(primes + prime_powers + others)
    if popcount(m2) > best_size:
        best_mask = m2
        best_size = popcount(m2)
        
    # Order 3: Random Shuffles
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 30: break
        random.shuffle(numbers)
        m = greedy_construct(numbers)
        if popcount(m) > best_size:
            best_mask = m
            best_size = popcount(m)

    # Simulated Annealing / Random Walk on Valid Sets
    # We will perform random moves: Add, Remove, Swap
    # Acceptance:
    # - Add: Always accept if valid (size +1)
    # - Remove: Always accept (size -1) - allows escaping
    # - Swap: Accept if valid (size 0)
    # To ensure we don't just shrink forever, we bias towards Add.
    # But we need to escape local maxima.
    
    current_mask = best_mask
    current_size = best_size
    
    # Temperature schedule
    # We don't strictly need T if we use the "always accept removal" strategy,
    # but we can use T to control the probability of accepting a "bad" swap?
    # Swaps don't change size, so they are neutral.
    # The main escape mechanism is Removal.
    
    # Let's run a loop until deadline
    t_start = time.time()
    last_improve = t_start
    
    while time.time() < deadline - 2:
        # Choose move type
        # 60% Add, 20% Swap, 20% Remove
        move_type = random.random()
        
        selected = get_selected(current_mask)
        unselected = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
        
        if not selected or not unselected:
            break
            
        if move_type < 0.6:
            # Try Add
            x = random.choice(unselected)
            if can_add(x, current_mask):
                current_mask |= (1 << x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    last_improve = time.time()
                    
        elif move_type < 0.8:
            # Try Swap
            # Pick x from selected, y from unselected
            x = random.choice(selected)
            y = random.choice(unselected)
            
            # Temporarily remove x
            temp_mask = current_mask & ~(1 << x)
            if can_add(y, temp_mask):
                current_mask = temp_mask | (1 << y)
                # Size unchanged
                
        else:
            # Try Remove
            x = random.choice(selected)
            current_mask &= ~(1 << x)
            current_size -= 1
            
        # If we haven't improved in a long time, we might be stuck.
        # The random walk should handle it.
        
    # Final Polish: Greedy add on best_mask
    # Try to add any remaining numbers
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not (best_mask & (1 << x)):
                if can_add(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True

    return get_selected(best_mask)
