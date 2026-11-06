import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    max_val = 512
    all_nums = list(range(1, max_val + 1))
    
    # Pre-compute conflict scores (number of 3-APs centered at x)
    # This is a cheap proxy for how "crowded" a number is.
    # Lower score = safer to include.
    scores = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        # Count pairs (a, b) such that a < x < b and a+b=2x
        # a ranges from max(1, 2x-512) to x-1
        start_a = max(1, 2 * x - max_val)
        end_a = x - 1
        if start_a <= end_a:
            scores[x] = end_a - start_a + 1

    # Order by increasing conflict score (safe first)
    order_safe = sorted(all_nums, key=lambda x: scores[x])
    
    # Also try reverse (dangerous first) and random
    order_unsafe = sorted(all_nums, key=lambda x: -scores[x])
    
    best_set = []
    best_len = 0
    
    rng = random.Random(42)

    # Helper: Fast check for 3-term AP in a set (subset size 2)
    def has_3ap(S):
        s_set = set(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                # Check if (a+b)/2 is in S
                # a+b must be even
                if (a + b) % 2 == 0:
                    m = (a + b) // 2
                    if m in s_set and m != a and m != b:
                        return True
        return False

    # Helper: Full DP check for validity
    # Returns True if valid
    def is_valid(S):
        n = len(S)
        if n < 3:
            return True
        
        s_set = set(S)
        
        # For each element a in S, check if it is the mean of any subset of S\{a}
        for a in S:
            others = [x for x in S if x != a]
            m = len(others)
            
            # DP for subset sums of 'others'
            # dp[k] is a bitset of achievable sums with k elements
            # Max sum is sum(others)
            # We only care if k*a is achievable for k>=2
            
            # Optimization: We can just check if k*a is possible.
            # But standard DP builds all sums.
            # Since max_val is 512 and n is small (<50), sum is < 25600.
            # Bitset DP is fast.
            
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
            
            # Check violations
            for k in range(2, m + 1):
                target = k * a
                if (dp[k] >> target) & 1:
                    return False
        return True

    # Helper: Count conflicts (number of violations) for hill climbing
    def count_conflicts(S):
        n = len(S)
        if n < 3:
            return 0
        total = 0
        for a in S:
            others = [x for x in S if x != a]
            m = len(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
            for k in range(2, m + 1):
                target = k * a
                if (dp[k] >> target) & 1:
                    total += 1
        return total

    # Strategy 1: Greedy construction with multiple orders
    for order in [order_safe, order_unsafe]:
        if time.time() > deadline - 10:
            break
        
        current = []
        for x in order:
            if time.time() > deadline - 5:
                break
            # Fast check: 3AP
            if has_3ap(current + [x]):
                continue
            # Full check
            if is_valid(current + [x]):
                current.append(x)
                if len(current) >= 48:
                    break
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current[:]
            
        if best_len >= 48:
            return best_set

    # Strategy 2: Local Search / Hill Climbing
    # Start from the best set found
    if not best_set:
        # If greedy failed completely, start with a random valid set
        # Try to build a small valid set first
        current = []
        for x in order_safe:
            if is_valid(current + [x]):
                current.append(x)
            if len(current) >= 10:
                break
        best_set = current
        best_len = len(best_set)
    
    S_set = set(best_set)
    complement = [x for x in all_nums if x not in S_set]
    
    current_conflicts = count_conflicts(best_set)
    
    # Local Search Loop
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2:
        if len(best_set) >= 48:
            break
        
        if no_improve > max_no_improve:
            # Random restart or perturbation
            # Try to add a random element even if it creates conflicts, then fix?
            # Or just reset to best and try a different random seed?
            # For simplicity, just break and try a new greedy phase?
            # Let's try to add a random element from complement to break the plateau
            if complement:
                y = rng.choice(complement)
                new_S = best_set + [y]
                new_conf = count_conflicts(new_S)
                # Accept if it doesn't increase conflicts too much?
                # Or just accept to move?
                # Let's just accept to move and see if we can fix it.
                best_set = new_S
                S_set.add(y)
                complement.remove(y)
                current_conflicts = new_conf
                no_improve = 0
                continue
            else:
                break
                
        # 1. Try to add an element (increase size)
        if complement:
            # Pick a few random candidates from complement
            # Prefer those with low conflict scores
            candidates = sorted(complement, key=lambda x: scores[x])[:20]
            rng.shuffle(candidates)
            
            added = False
            for y in candidates:
                if time.time() > deadline - 2:
                    break
                new_S = best_set + [y]
                # Fast 3AP check
                if has_3ap(new_S):
                    continue
                # Full check
                if is_valid(new_S):
                    best_set = new_S
                    S_set.add(y)
                    complement.remove(y)
                    added = True
                    break
            
            if added:
                no_improve = 0
                continue
        
        # 2. Try to swap (maintain size, improve structure)
        if not complement:
            break
            
        x = rng.choice(best_set)
        y = rng.choice(complement)
        
        if x == y:
            continue
            
        new_S = [z for z in best_set if z != x] + [y]
        
        # Check if valid
        if is_valid(new_S):
            # Valid swap: we can try to add more later
            best_set = new_S
            S_set.remove(x)
            S_set.add(y)
            complement.remove(y)
            complement.append(x)
            current_conflicts = 0
            no_improve = 0
            continue
        
        # Check if fewer conflicts
        new_conf = count_conflicts(new_S)
        if new_conf < current_conflicts:
            best_set = new_S
            S_set.remove(x)
            S_set.add(y)
            complement.remove(y)
            complement.append(x)
            current_conflicts = new_conf
            no_improve = 0
        else:
            no_improve += 1

    return best_set

# Test block (not executed in evaluation)
if __name__ == "__main__":
    s = construct()
    print(f"Size: {len(s)}")
    print(s)
