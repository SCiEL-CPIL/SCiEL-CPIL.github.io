# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_CARD = 48

def _is_valid_add_fast(c, S, S_set):
    """
    Check if c can be added to set S (list of ints) and S_set (set of ints).
    1. c must not divide sum of any 3 distinct elements in S.
    2. For each a in S, a must not divide (c + x + y) for any distinct x, y in S\{a}.
    """
    n = len(S)
    
    # Check 1: c divides x+y+z for x,y,z in S, x!=y!=z
    # We need to check if there exist x, y, z in S distinct such that (x+y+z) % c == 0
    # Since S is small (<=48), we can use a set of residues or just pair sums.
    # For each pair (x, y), check if z = k*c - x - y is in S and distinct.
    # However, z must be in [1, 192].
    # Let's iterate pairs and check membership.
    # To avoid O(n^3), we can precompute a dictionary of residues for S mod c?
    # No, c changes.
    
    # Optimization: If c is large, the number of multiples is small.
    # Sum of 3 distinct elements from S is between min_sum and max_sum.
    # Let's just do the pair loop and set lookup.
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            # We need z in S, z != x, z != y, such that (x + y + z) % c == 0
            # z = m*c - x - y
            # Range of z: 1 to 192.
            # m*c - x - y must be in [1, 192]
            # So m*c in [1 + x + y, 192 + x + y]
            # Let's find possible m values.
            min_val = 1 + x + y
            max_val = 192 + x + y
            
            # m is integer such that m*c is in [min_val, max_val]
            # m_min = ceil(min_val / c)
            # m_max = floor(max_val / c)
            
            m_min = (min_val + c - 1) // c
            m_max = max_val // c
            
            for m in range(m_min, m_max + 1):
                z = m * c - x - y
                if z in S_set and z != x and z != y:
                    return False
                    
    # Check 2: For each a in S, check if a divides c + x + y for x, y in S\{a}
    # x, y distinct, both != a
    for idx_a, a in enumerate(S):
        # We need x, y in S\{a}, x != y such that (c + x + y) % a == 0
        # c + x + y = k * a
        # x + y = k * a - c
        # x, y in S\{a}
        
        # Iterate over possible k?
        # Max x+y is 192+191=383.
        # Min x+y is 1+2=3.
        # So k*a - c is in [3, 383].
        # k*a is in [c+3, c+383].
        
        min_val = c + 3
        max_val = c + 383
        
        k_min = (min_val + a - 1) // a
        k_max = max_val // a
        
        # Create a set of S excluding a for faster lookup?
        # S_minus_a = S_set - {a}
        
        for k in range(k_min, k_max + 1):
            target_sum = k * a - c
            # We need x, y in S\{a}, x != y, x + y = target_sum
            # Iterate x in S\{a}
            for x in S:
                if x == a:
                    continue
                y = target_sum - x
                # Check if y is in S\{a}, y != x
                if y in S_set and y != a and y != x:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending Greedy with fast check
    # Bias towards larger numbers as they are less likely to divide sums of others
    # (since sums are larger, and divisibility by large numbers is rarer for fixed range sums)
    # Actually, if c is large, c | S implies S is a multiple of c.
    # If we pick all numbers > 100, sums are 300-576.
    # Multiples of c (100-192) in this range are 2c, 3c.
    # So we just avoid x+y+z = 2c or 3c.
    
    candidates = list(range(N, 0, -1))
    selected = []
    S_set = set()
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_CARD:
            break
            
        if _is_valid_add_fast(c, selected, S_set):
            selected.append(c)
            S_set.add(c)
            
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)
        
    # Strategy 2: Randomized Restart with Biased Ordering
    # Try mixing in smaller numbers if we have space, or just try different random orders
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # Create a candidate list with some randomness
        # We want to try to get 48 elements.
        # Large numbers are "safer" in terms of not dividing sums?
        # Let's try a mix: 70% large, 30% random
        cand = list(range(1, N + 1))
        # Shuffle
        rng.shuffle(cand)
        
        # Or sort by a heuristic score
        # Let's just try random restarts with the fast checker
        # The fast checker allows more iterations
        
        S = []
        S_set = set()
        
        # To encourage diversity, we can try to pick from the whole range
        # But descending worked well in seed. Let's try descending with small noise
        # Or just random shuffle
        # Let's try a few different biases
        
        if restart % 3 == 0:
            order = list(range(N, 0, -1))
        elif restart % 3 == 1:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
        for c in order:
            if len(S) >= MAX_CARD:
                break
            if c in S_set:
                continue
            if _is_valid_add_fast(c, S, S_set):
                S.append(c)
                S_set.add(c)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        # Local Search: Try to swap elements to free up space for more?
        # Or just try to add any remaining element
        if len(S) < MAX_CARD and time.time() < deadline - 2.0:
            remaining = [c for c in range(1, N + 1) if c not in S_set]
            rng.shuffle(remaining)
            for c in remaining:
                if len(S) >= MAX_CARD:
                    break
                if _is_valid_add_fast(c, S, S_set):
                    S.append(c)
                    S_set.add(c)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)

    return best_set
# EVOLVE-BLOCK-END
