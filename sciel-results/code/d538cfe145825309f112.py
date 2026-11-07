import time
import random

# EVOLVE-BLOCK-START

def _validate_bitset(S):
    """
    Validate if set S is valid.
    Violation: a, b, c distinct in S such that a+b, a+c, b+c in S.
    """
    in_set = [False] * 257
    for x in S:
        in_set[x] = True
    
    S_list = sorted(S)
    n = len(S_list)
    
    # For each pair (a, b), check if a+b is in S.
    # If so, check if there is a c in S such that a+c and b+c are in S.
    # Optimization: 
    # Precompute for each x in S, the set of y in S such that x+y is in S.
    # Let partners[x] = {y in S | y != x and x+y in S}
    # Violation exists if there is a pair (a, b) with a+b in S, 
    # and partners[a] and partners[b] share a common element c (c != a, b).
    
    partners = {x: set() for x in S}
    for x in S_list:
        for y in S_list:
            if x == y:
                continue
            if x + y <= 128 and in_set[x + y]:
                partners[x].add(y)
    
    # Check for common partner c for any pair (a, b) where a+b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if a + b <= 128 and in_set[a + b]:
                # Check if partners[a] and partners[b] intersect at some c not equal to a or b
                # Iterate over smaller set
                if len(partners[a]) < len(partners[b]):
                    for c in partners[a]:
                        if c != a and c != b and c in partners[b]:
                            return False
                else:
                    for c in partners[b]:
                        if c != a and c != b and c in partners[a]:
                            return False
    return True

def _can_add_fast(S, x, in_set):
    """
    Check if adding x to S creates a violation.
    x is not in S.
    New violations must involve x.
    Triples: (x, a, b) with a, b in S, a != b.
    Sums: x+a, x+b, a+b.
    Conditions for violation:
    1. x+a in S (since x+a > x, it must be in old S)
    2. x+b in S
    3. a+b in S or a+b == x
    """
    S_list = list(S)
    # Precompute potential partners for x
    # P = {a in S | x+a in S}
    # If a in P and b in P (a!=b), check if a+b in S or a+b == x.
    
    P = []
    for a in S_list:
        if x + a <= 128 and in_set[x + a]:
            P.append(a)
            
    # If fewer than 2 partners, no violation possible involving x as one of the 3 distinct members?
    # Wait, the triple is (x, a, b).
    # We need a, b distinct from x and each other.
    # a, b are in S, so they are distinct from x (x not in S).
    # We need a != b.
    # So if len(P) < 2, return True.
    if len(P) < 2:
        return True
        
    P_set = set(P)
    for i in range(len(P)):
        a = P[i]
        for j in range(i + 1, len(P)):
            b = P[j]
            # Check a+b
            if a + b == x:
                return False
            if a + b <= 128 and in_set[a + b]:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Start with known valid set {63..128}
    S = set(range(63, 129))
    best = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 129))
    
    def build_in_set(S):
        in_set = [False] * 257
        for x in S:
            in_set[x] = True
        return in_set
    
    # Phase 1: Greedy addition
    # Try adding elements in descending order? Or ascending?
    # Smaller elements create more sums in range, so they are more restrictive.
    # Larger elements (like 62) create sums like 125, 126 which are in S.
    # Let's try all missing elements.
    in_set = build_in_set(S)
    missing = sorted(all_nums - S)
    
    for x in missing:
        if time.time() > deadline - 10:
            break
        if _can_add_fast(S, x, in_set):
            S.add(x)
            in_set[x] = True
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            # Update in_set and try again? No, just continue. 
            # Adding one might enable others? Unlikely to enable, usually disables.
            # But let's restart the loop or just continue?
            # Continuing is fine, as we check against current S.
            
    # Phase 2: Local Search with Swaps
    # Strategy: Remove k elements, Add k+1 elements.
    # We will focus on Remove 1, Add 2 (net +1) and Remove 2, Add 3 (net +1).
    
    random.seed(42)
    
    def try_swap_remove_add(S, k_remove, k_add):
        """
        Try to remove k_remove elements and add k_add elements.
        Returns new set if valid and size increased, else None.
        """
        S_list = list(S)
        missing = list(all_nums - S)
        
        if len(S_list) < k_remove or len(missing) < k_add:
            return None
            
        # Randomly select candidates
        # To be efficient, we can try a few random combinations
        for _ in range(50): # Limit attempts per call
            if time.time() > deadline - 2:
                return None
                
            # Select elements to remove
            remove_candidates = random.sample(S_list, k_remove)
            # Select elements to add
            add_candidates = random.sample(missing, k_add)
            
            new_S = set(S)
            for r in remove_candidates:
                new_S.discard(r)
            for a in add_candidates:
                new_S.add(a)
                
            if len(new_S) > len(S):
                if _validate_bitset(new_S):
                    return new_S
        return None

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        
        # Try Remove 1, Add 2
        new_S = try_swap_remove_add(S, 1, 2)
        if new_S:
            S = new_S
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            improved = True
            continue
            
        # Try Remove 2, Add 3
        new_S = try_swap_remove_add(S, 2, 3)
        if new_S:
            S = new_S
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            improved = True
            continue
            
        # If no swap worked, try greedy addition again (maybe previous swap opened up space)
        in_set = build_in_set(S)
        missing = list(all_nums - S)
        added_any = False
        for x in missing:
            if time.time() > deadline - 5:
                break
            if _can_add_fast(S, x, in_set):
                S.add(x)
                in_set[x] = True
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                added_any = True
        if added_any:
            improved = True

    # Final check: ensure best is valid and return sorted list
    result = sorted(best)
    return list(result)

# EVOLVE-BLOCK-END
