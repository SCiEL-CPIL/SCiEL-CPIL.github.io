import time
import random

# EVOLVE-BLOCK-START

def _validate_bitset(S):
    """
    Validate if set S is valid.
    Violation: a, b, c distinct in S such that a+b, a+c, b+c in S.
    Returns True if valid, False otherwise.
    """
    in_set = [False] * 257
    for x in S:
        in_set[x] = True
    
    S_list = sorted(S)
    n = len(S_list)
    
    # Precompute partners: for each x, set of y in S such that x+y in S
    # This helps in checking if a common partner exists.
    # However, constructing this dict every time is O(N^2).
    # For N=70, it's fine.
    
    # Optimization: 
    # A violation exists if there are distinct a, b, c in S 
    # such that a+b, a+c, b+c are in S.
    # This implies:
    # 1. a+b in S
    # 2. a+c in S
    # 3. b+c in S
    
    # Let's iterate over pairs (a, b) with a < b.
    # If a+b is in S, then we need to check if there exists c != a, b
    # such that a+c in S AND b+c in S.
    
    # For a fixed pair (a, b), the candidates for c are:
    # C = { c in S | c != a, c != b, a+c in S, b+c in S }
    # If C is not empty, violation.
    
    # To make this fast:
    # For each a, precompute valid_adds_a = { y in S | a+y in S }
    # For each b, precompute valid_adds_b = { y in S | b+y in S }
    # Then intersection of valid_adds_a and valid_adds_b (excluding a, b) must be empty.
    
    # Building valid_adds is O(N^2) total for all a.
    # Checking intersection is O(N) per pair.
    # Total O(N^3) in worst case? No.
    # valid_adds_a is a set. Intersection of two sets of size K is O(K).
    # Number of pairs is N^2/2.
    # So O(N^3) if sets are large. But sets are sparse?
    # In {63..128}, valid_adds for 63 is {65} (128). Size 1.
    # For 1, valid_adds is huge.
    
    # Let's stick to the previous partner intersection logic which was efficient enough.
    
    partners = {x: set() for x in S}
    for x in S_list:
        for y in S_list:
            if x == y:
                continue
            if x + y <= 128 and in_set[x + y]:
                partners[x].add(y)
    
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
    """
    S_list = list(S)
    P = []
    for a in S_list:
        if x + a <= 128 and in_set[x + a]:
            P.append(a)
            
    if len(P) < 2:
        return True
        
    P_set = set(P)
    for i in range(len(P)):
        a = P[i]
        for j in range(i + 1, len(P)):
            b = P[j]
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
    # Try adding elements in descending order? 
    # Larger elements are "safer" because their sums are more likely to exceed 128.
    in_set = build_in_set(S)
    missing = sorted(all_nums - S, reverse=True)
    
    for x in missing:
        if time.time() > deadline - 20:
            break
        if _can_add_fast(S, x, in_set):
            S.add(x)
            in_set[x] = True
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
    
    # Phase 2: Systematic Single-Swap Hill Climbing
    # Try to replace one element with another to maintain size (neutral) or increase?
    # Single swap maintains size. We need net gain.
    # So we rely on Greedy Add for gains.
    # But maybe removing a "bad" element allows adding TWO "good" elements?
    # This is the 1-for-2 swap.
    
    # Let's do a more aggressive local search:
    # 1. Try all single additions (already done).
    # 2. Try all 1-for-2 swaps: Remove u, Add v1, Add v2.
    #    This is expensive: O(N * M^2 * Validate).
    #    N=68, M=60. 68 * 3600 * 5000 ops ~ 1 billion. Too slow.
    
    # Heuristic 1-for-2:
    # Identify "restrictive" elements.
    # An element u is restrictive if it participates in many "potential" violations.
    # Or simply:
    # For each u in S:
    #   S_temp = S - {u}
    #   Try to add as many elements from Missing as possible to S_temp greedily.
    #   If size(S_temp + added) > size(S), accept.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = sorted(S)
        missing = list(all_nums - S)
        
        # Try removing each element u
        for u in S_list:
            if time.time() > deadline - 5:
                break
            
            S_temp = set(S)
            S_temp.discard(u)
            in_set_temp = build_in_set(S_temp)
            
            # Greedily add elements to S_temp
            # Sort missing by "ease" of addition? 
            # Let's just try all missing in descending order (safer).
            added_count = 0
            for v in sorted(missing, reverse=True):
                if time.time() > deadline - 5:
                    break
                if _can_add_fast(S_temp, v, in_set_temp):
                    S_temp.add(v)
                    in_set_temp[v] = True
                    added_count += 1
            
            # If we added at least 2 elements (net gain +1) or even 1 (net gain 0, but maybe different structure helps later?)
            # We want net gain > 0. So added_count >= 2.
            if added_count >= 2:
                if len(S_temp) > len(S):
                    S = S_temp
                    if len(S) > best_size:
                        best = set(S)
                        best_size = len(S)
                    improved = True
                    break # Restart the while loop with new S
    
    # Phase 3: Random 1-for-2 and 2-for-3 swaps (if time remains)
    # Use random sampling to avoid full O(N*M^2) search
    random.seed(42)
    
    def try_random_swap(S, k_remove, k_add):
        S_list = list(S)
        missing = list(all_nums - S)
        
        if len(S_list) < k_remove or len(missing) < k_add:
            return None
            
        # Try a few random combinations
        for _ in range(100):
            if time.time() > deadline - 2:
                return None
                
            remove_candidates = random.sample(S_list, k_remove)
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

    while time.time() < deadline - 2:
        new_S = try_random_swap(S, 1, 2)
        if new_S:
            S = new_S
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            continue
        
        new_S = try_random_swap(S, 2, 3)
        if new_S:
            S = new_S
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            continue
            
        # If no random swap worked, break to save time for final validation
        break

    # Final check
    result = sorted(best)
    # Ensure types are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
