import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s
    n = 512

    # Precompute all 4-term arithmetic progressions
    # For each element x, store the list of "other 3" tuples for each AP containing x
    ap_others = {i: [] for i in range(1, n + 1)}
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_others[a].append((b, c, e))
            ap_others[b].append((a, c, e))
            ap_others[c].append((a, b, e))
            ap_others[e].append((a, b, c))
    
    # Precompute conflict pairs: for each x, the set of y such that x and y 
    # appear together in some 4-AP with two other elements
    # This lets us check: can we add x if no conflict pair (x,y) has y in S?
    # Actually, we need: adding x creates a 4-AP iff there exist y,z,w in S 
    # such that {x,y,z,w} forms a 4-AP.
    # Precompute for each x: list of (y, z, w) triples from ap_others[x]
    # can_add(x, S) = for all (y,z,w) in ap_others[x], NOT (y in S AND z in S AND w in S)
    
    # For speed, precompute the "conflict sets" differently:
    # For each x, and for each AP (x, y, z, w), the "blocking set" is {y, z, w}.
    # We can precompute for each x a list of frozensets or use numpy.
    
    # Use numpy bitset representation
    def set_to_bits(S):
        bits = np.zeros(n + 1, dtype=bool)
        for x in S:
            bits[x] = True
        return bits
    
    def can_add(x, S_bits):
        """Check if adding x to S creates a 4-AP."""
        for other in ap_others[x]:
            if S_bits[other[0]] and S_bits[other[1]] and S_bits[other[2]]:
                return False
        return True
    
    def has_4ap(S):
        """Verify if S contains a 4-AP."""
        S_set = set(S)
        for a in S_set:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S_set and (a + 2 * d) in S_set and (a + 3 * d) in S_set:
                    return True
        return False

    def ternary_digits_set(n, allowed_digits):
        """Generate all numbers in [1, n] whose base-3 representation uses only digits in allowed_digits."""
        result = set()
        # 3^6 = 729 > 512, so we need at most 6 digits
        for mask in range(1, 1 << 6):
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    # We need to assign a digit from allowed_digits to this position
                    # But mask only tells us which positions are "active"
                    # This approach doesn't work for multiple digit choices
                    pass
            # Better: enumerate all numbers and check their base-3 digits
        for num in range(1, n + 1):
            temp = num
            valid = True
            while temp > 0:
                digit = temp % 3
                if digit not in allowed_digits:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(num)
        return result

    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        S = set(S_init)
        S_bits = set_to_bits(S)
        for x in order:
            if x in S:
                continue
            if can_add(x, S_bits):
                S.add(x)
                S_bits[x] = True
        return S

    def local_search(S, deadline, max_rounds=1000):
        """
        Local search with 1-2 moves and 1-1 swaps.
        """
        S = set(S)
        S_bits = set_to_bits(S)
        current_size = len(S)
        no_improvement_count = 0
        all_elements = list(range(1, n + 1))
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            if not S:
                break
            
            # Try 1-2 move: remove 1, add 2
            to_remove = random.choice(list(S))
            S_minus_bits = S_bits.copy()
            S_minus_bits[to_remove] = False
            
            # Find candidates that can be added to S_minus
            valid_candidates = []
            for c in all_elements:
                if S_minus_bits[c]:
                    continue
                if can_add(c, S_minus_bits):
                    valid_candidates.append(c)
            
            if len(valid_candidates) >= 2:
                random.shuffle(valid_candidates)
                found_pair = False
                temp_bits = S_minus_bits.copy()
                for i in range(min(len(valid_candidates), 50)):
                    c1 = valid_candidates[i]
                    if temp_bits[c1]:
                        continue
                    if not can_add(c1, temp_bits):
                        continue
                    temp_bits[c1] = True
                    for j in range(i + 1, min(i + 51, len(valid_candidates))):
                        c2 = valid_candidates[j]
                        if temp_bits[c2]:
                            continue
                        if can_add(c2, temp_bits):
                            S = S - {to_remove} | {c1, c2}
                            S_bits = temp_bits
                            S_bits[c2] = True
                            current_size = len(S)
                            no_improvement_count = 0
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if not found_pair:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 40:
                # Try 1-1 swap to diversify
                to_remove = random.choice(list(S))
                S_minus_bits = S_bits.copy()
                S_minus_bits[to_remove] = False
                
                found_swap = False
                for c in all_elements:
                    if S_minus_bits[c]:
                        continue
                    if can_add(c, S_minus_bits):
                        S = S - {to_remove} | {c}
                        S_bits = S_minus_bits
                        S_bits[c] = True
                        found_swap = True
                        break
                
                if not found_swap:
                    no_improvement_count += 1
                else:
                    no_improvement_count = max(0, no_improvement_count - 15)
                
                if no_improvement_count > 100:
                    break
        
        return S

    def kick_and_repair(S, deadline):
        """Random kick: remove a few random elements, then repair greedily."""
        S = set(S)
        if not S:
            return S
        
        kick_size = random.randint(1, 4)
        to_remove = random.sample(list(S), min(kick_size, len(S)))
        S_kicked = S - set(to_remove)
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_repaired = greedy_construct(S_kicked, order)
        
        S_final = local_search(S_repaired, deadline, max_rounds=150)
        return S_final

    # Multiple ternary-based initializations
    random.seed(42)
    
    # Try different digit sets
    digit_sets = [
        {0, 1},   # classic
        {1, 2},   # complement-like
        {0, 2},   # another variant
    ]
    
    best_set = set()
    best_size = 0
    
    for digits in digit_sets:
        init_set = ternary_digits_set(n, digits)
        order = list(range(1, n + 1))
        random.shuffle(order)
        greedy_result = greedy_construct(init_set, order)
        if len(greedy_result) > best_size:
            best_set = greedy_result
            best_size = len(best_set)
    
    # Also try greedy from empty
    order = list(range(1, n + 1))
    random.shuffle(order)
    greedy_empty = greedy_construct(set(), order)
    if len(greedy_empty) > best_size:
        best_set = greedy_empty
        best_size = len(best_set)
    
    # Local search on best
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=2000)
        best_size = len(best_set)
    
    # Multiple restarts with kicks
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        # Vary the initialization
        if attempt % 4 == 0:
            S_init = set()
        elif attempt % 4 == 1:
            S_init = ternary_digits_set(n, {0, 1})
        elif attempt % 4 == 2:
            S_init = ternary_digits_set(n, {1, 2})
        else:
            S_init = ternary_digits_set(n, {0, 2})
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        current = greedy_construct(S_init, order)
        
        if len(current) >= best_size:
            current = local_search(current, deadline, max_rounds=1000)
        else:
            current = kick_and_repair(current, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked = kick_and_repair(best_set, deadline)
        if len(kicked) > best_size:
            best_set = kicked
            best_size = len(best_set)
    
    # Final local search
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=1000)
        best_size = len(best_set)
    
    # Final verification
    if has_4ap(best_set):
        # Fallback to safe ternary set
        best_set = ternary_digits_set(n, {0, 1})
    
    return sorted(best_set)
