import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N + 1))
    
    # Precompute conflict pairs for the reciprocal equation 1/a = 1/b + 1/c
    # which is equivalent to (b-a)(c-a) = a^2
    # For three distinct elements a, b, c in S, we need 1/a = 1/b + 1/c
    # WLOG assume a < b < c, then 1/a > 1/b > 1/c, so 1/a = 1/b + 1/c is possible
    # Actually, let's think about this more carefully.
    # 1/a = 1/b + 1/c means a = bc/(b+c), so a < b and a < c.
    # So a is the smallest of the three.
    # Given a, we need b, c > a with 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2.
    # So for a given a, the pairs (b, c) with b > a, c > a, b != c, and (b-a)(c-a) = a^2
    # are determined by divisors of a^2.
    
    # For each a, find all valid (b, c) pairs
    conflicts = {}  # conflicts[a] = set of (b, c) pairs that would form a triple with a
    for a in range(1, N + 1):
        pairs = []
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    pairs.append((b, c))
        conflicts[a] = pairs
    
    # For local search, we need to check if adding x to S creates a conflict.
    # A conflict occurs if there exist a, b in S such that (x, a, b) forms a triple.
    # That is, if x is the smallest: need a, b in S with (a-x)(b-x) = x^2
    # If x is the largest: need a, b in S with a < x, b < x, and 1/a = 1/b + 1/x or 1/b = 1/a + 1/x
    #   i.e., (b-a)(x-a) = a^2 or (a-b)(x-b) = b^2
    # If x is in the middle: need a, b in S with a < x < b and 1/a = 1/x + 1/b
    #   i.e., (x-a)(b-a) = a^2
    
    # Let's precompute for each x, what pairs (y, z) in S would conflict with x.
    # Actually, it's easier to just check: for each pair (y, z) in S, does x complete a triple?
    # But that's O(|S|^2) per check.
    
    # Better approach: for each x, precompute all (y, z) such that {x, y, z} is a reciprocal triple.
    # Then x can be added to S iff no such (y, z) both in S.
    
    def get_conflict_pairs(x):
        """Return list of (y, z) such that {x, y, z} forms a reciprocal triple."""
        result = []
        # Case 1: x is smallest. Need y, z > x with (y-x)(z-x) = x^2
        x2 = x * x
        for d in range(1, int(x2**0.5) + 1):
            if x2 % d == 0:
                e = x2 // d
                y = x + d
                z = x + e
                if y <= N and z <= N and y != z:
                    result.append((y, z))
        # Case 2: x is largest. Need y, z < x with 1/y = 1/z + 1/x or 1/z = 1/y + 1/x
        # 1/y = 1/z + 1/x => (z-y)(x-y) = y^2
        for y in range(1, x):
            # Check if there exists z < x such that (z-y)(x-y) = y^2
            # z = y + y^2/(x-y)
            denom = x - y
            if denom <= 0:
                continue
            if (y * y) % denom == 0:
                z = y + (y * y) // denom
                if 1 <= z <= N and z != y and z < x:
                    result.append((y, z))
        # Case 3: x is middle. Need y < x < z with 1/y = 1/x + 1/z
        # (x-y)(z-y) = y^2
        for y in range(1, x):
            denom = x - y
            if denom <= 0:
                continue
            if (y * y) % denom == 0:
                z = y + (y * y) // denom
                if z > x and z <= N and z != x:
                    result.append((y, z))
        return result
    
    # Precompute conflict pairs for all x
    all_conflict_pairs = [None] + [get_conflict_pairs(x) for x in range(1, N + 1)]
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a reciprocal triple."""
        for (y, z) in all_conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
        return True
    
    def count_conflicts(x, S_set):
        """Count how many conflict pairs (y, z) both in S_set for x."""
        cnt = 0
        for (y, z) in all_conflict_pairs[x]:
            if y in S_set and z in S_set:
                cnt += 1
        return cnt
    
    # Greedy initialization: add elements in random order, keep if no conflict
    def greedy_init():
        S_set = set()
        order = elements[:]
        random.shuffle(order)
        for x in order:
            if can_add(x, S_set):
                S_set.add(x)
        return S_set
    
    # Local search with 1-swap and 2-swap moves
    def local_search(S_set, max_iters=50000):
        S_list = sorted(S_set)
        best_size = len(S_set)
        no_improve = 0
        iters = 0
        while iters < max_iters and time.time() < deadline - 2.0:
            iters += 1
            improved = False
            
            # Try 1-swap: remove u from S, add x not in S
            # Net change in size: 0 if we just swap, but we want to increase size
            # Actually, 1-swap means remove one and add one, so size stays same.
            # To increase size, we need to add x without removing anything, but that requires can_add(x, S).
            # Let's first try to add any x not in S.
            for x in elements:
                if x not in S_set:
                    if can_add(x, S_set):
                        S_set.add(x)
                        best_size = max(best_size, len(S_set))
                        improved = True
                        break
            
            if improved:
                no_improve = 0
                continue
            
            # Try 2-swap: remove u, v from S, add x, y not in S
            # This can increase size by 0, but might allow further additions.
            # Actually, let's try: remove u from S, then try to add x not in S.
            # If that works, we've effectively swapped u for x, size same.
            # But if after removing u, we can add TWO elements, that's a win.
            
            # Let's try: for each u in S, remove u, then greedily try to add elements.
            for u in list(S_set):
                if time.time() > deadline - 2.0:
                    break
                S_set.remove(u)
                added = []
                for x in elements:
                    if x not in S_set and x != u:
                        if can_add(x, S_set):
                            S_set.add(x)
                            added.append(x)
                if len(added) > 1:
                    # We removed 1 and added >1, net gain
                    best_size = max(best_size, len(S_set))
                    improved = True
                    break
                else:
                    # Revert
                    S_set.discard(u)
                    for x in added:
                        S_set.discard(x)
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 200:
                    break
        
        return S_set
    
    # Run multiple restarts
    best_S = set()
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        S_set = greedy_init()
        S_set = local_search(S_set, max_iters=10000)
        
        if len(S_set) > len(best_S):
            best_S = S_set.copy()
            if len(best_S) >= 150:  # likely optimal or near-optimal
                break
    
    return list(best_S)
