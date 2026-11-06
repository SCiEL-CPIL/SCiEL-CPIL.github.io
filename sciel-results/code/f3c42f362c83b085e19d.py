import random
import time

def construct():
    N = 1000
    max_sum = 2 * N
    
    # Bitsets for O(1) validity checks
    # S_bits: bit i is 1 if i is in S
    # BadSums: bit k is 1 if counts[k] == 2
    
    best_S = set()
    best_size = 0
    start_time = time.time()
    time_limit = 115

    def greedy_random(order):
        S = set()
        S_bits = 0
        counts = [0] * (max_sum + 1)
        BadSums = 0
        
        for x in order:
            # Check validity using bitsets
            # 1. Check 2x
            if counts[2 * x] >= 2:
                continue
            
            # 2. Check x + y for y in S
            # We need to check if any y in S has counts[x+y] == 2
            # This is equivalent to checking if S_bits intersects with (BadSums >> x)
            # Note: BadSums has bit k set if counts[k]==2.
            # We want y such that x+y is in BadSums.
            # So y is in BadSums shifted right by x.
            # But we also need y to be in S.
            # So (S_bits & (BadSums >> x)) should be 0.
            
            # Also need to ensure x is not already in S (it isn't, we iterate unique candidates)
            
            if (S_bits & (BadSums >> x)) != 0:
                continue
            
            # Valid, add x
            S.add(x)
            S_bits |= (1 << x)
            
            # Update counts and BadSums
            # 1. 2x
            counts[2 * x] += 1
            if counts[2 * x] == 2:
                BadSums |= (1 << (2 * x))
            
            # 2. x + y for all y in S (excluding x, but x was just added, so iterate old S? 
            # No, we iterate current S which includes x? 
            # Wait, if we iterate S after adding x, we will process (x,x) again?
            # The pair (x,x) is handled by the 2x block.
            # The pairs (x,y) for y in S_old are handled here.
            # So we should iterate over S_old.
            # Since we added x to S, let's iterate over list(S) before adding? 
            # Or just iterate over S and skip x?
            # If we iterate over S (which now contains x), we will see y=x.
            # x+x is 2x. We already updated counts[2x].
            # So we should skip x in the loop.
            
            for y in S:
                if y == x:
                    continue
                s = x + y
                counts[s] += 1
                if counts[s] == 2:
                    BadSums |= (1 << s)
        
        return S, len(S)

    # Phase 1: Greedy with random orders
    for i in range(20):
        if time.time() - start_time > time_limit - 20: # Leave time for local search
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S, size = greedy_random(candidates)
        if size > best_size:
            best_size = size
            best_S = set(S)

    # Phase 2: Local Search on best_S
    S = set(best_S)
    S_bits = 0
    for x in S:
        S_bits |= (1 << x)
    
    counts = [0] * (max_sum + 1)
    BadSums = 0
    # Rebuild counts
    for a in S:
        counts[2 * a] += 1
        if counts[2 * a] == 2:
            BadSums |= (1 << (2 * a))
        for b in S:
            if a < b:
                s = a + b
                counts[s] += 1
                if counts[s] == 2:
                    BadSums |= (1 << s)

    def can_add(x):
        if x in S:
            return False
        if counts[2 * x] >= 2:
            return False
        if (S_bits & (BadSums >> x)) != 0:
            return False
        return True

    def add(x):
        nonlocal S_bits, BadSums
        S.add(x)
        S_bits |= (1 << x)
        
        counts[2 * x] += 1
        if counts[2 * x] == 2:
            BadSums |= (1 << (2 * x))
        
        for y in S:
            if y == x:
                continue
            s = x + y
            counts[s] += 1
            if counts[s] == 2:
                BadSums |= (1 << s)

    def remove(x):
        nonlocal S_bits, BadSums
        S.remove(x)
        S_bits &= ~(1 << x)
        
        counts[2 * x] -= 1
        if counts[2 * x] == 1:
            BadSums &= ~(1 << (2 * x))
        
        for y in S:
            if y == x:
                continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 1:
                BadSums &= ~(1 << s)

    # Local Search
    # Use a list of available candidates for faster random selection?
    # Or just random.randint and check membership.
    # With fast can_add, membership check is O(1).
    
    while time.time() - start_time < time_limit:
        # Try to add
        if len(S) < N:
            x = random.randint(1, N)
            if can_add(x):
                add(x)
                continue
        
        # Try to swap
        # Pick random y in S
        y = random.choice(list(S))
        # Pick random x not in S
        # To avoid infinite loop if S is full, we must ensure x != y
        x = random.randint(1, N)
        if x == y:
            continue
            
        # Check if swap is valid
        # 1. Remove y
        remove(y)
        
        # 2. Check if x can be added
        if can_add(x):
            add(x)
        else:
            # Restore y
            add(y)
            
    return list(S)
