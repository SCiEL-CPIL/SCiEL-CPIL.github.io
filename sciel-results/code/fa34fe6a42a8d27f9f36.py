import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s
    
    n = 192
    max_size = 48
    
    # Precompute all 3-combinations of indices for the universe [1..192]
    # But we'll work with actual values. Let's precompute all triples from 1..192
    # There are C(192,3) = 192*191*190/6 = 1,164,320 triples. That's a lot but manageable.
    # Actually, let's be smarter. For a given set S of size 48, we only care about triples within S.
    
    # Let's use a bitset representation for speed.
    # Universe: 1..192. We can map value v to bit index v-1.
    # A set S is represented as an integer where bit i is set if (i+1) is in S.
    
    # Precompute for each possible value a (1..192), and for each pair (b,c) where b,c < a or b,c > a etc.
    # Actually, the constraint is: for each a in S, a does NOT divide (b+c+d) for any distinct b,c,d in S\{a}.
    
    # Let's precompute a table: for each a in 1..192, and for each sum s that is a multiple of a,
    # we need to know if there exist 3 distinct elements in S\{a} that sum to s.
    
    # Alternative approach: For a fixed set S, check validity by:
    # For each a in S:
    #   Let T = S \ {a}
    #   Compute all possible sums of 3 distinct elements in T.
    #   Check if any of these sums is divisible by a.
    
    # To speed this up, we can precompute for each a, the set of "forbidden" sums.
    # Forbidden sums for a are multiples of a: a, 2a, 3a, ...
    # The minimum sum of 3 distinct elements from [1,192] is 1+2+3=6.
    # The maximum sum is 190+191+192=573.
    # So for a given a, we only care about multiples of a in [6, 573].
    
    # Let's precompute for each a in 1..192:
    #   forbidden_multiples[a] = sorted list of multiples of a in [6, 573]
    
    forbidden_multiples = {}
    for a in range(1, n + 1):
        multiples = []
        m = a
        while m <= 573:
            if m >= 6:
                multiples.append(m)
            m += a
        forbidden_multiples[a] = multiples
    
    # For a set S, to check if a divides sum of 3 others:
    #   We need to check if any forbidden multiple of a can be formed as sum of 3 distinct elements in S\{a}.
    
    # To make this fast, let's represent S as a sorted list.
    # For each a in S, we can use a meet-in-the-middle or just iterate.
    # Since |S| <= 48, |S\{a}| <= 47. C(47,3) = 16215. For 48 elements, that's 48*16215 ~ 778k checks.
    # Each check is O(1) if we precompute the set of all 3-sums.
    
    # Let's precompute all 3-sums for a set S.
    # Actually, let's just compute the set of all 3-sums from S, and for each a in S, check if any multiple of a (that is not using a itself in the sum) is in the set.
    # Wait, the 3-sums are from S\{a}, so we need to be careful.
    
    # Better approach: For each a in S, compute the set of 3-sums from S\{a}.
    # But that's expensive to recompute for each a.
    
    # Alternative: Compute all 3-sums from S. For each 3-sum s, it uses 3 elements {x,y,z}.
    # For each a in S\{x,y,z}, if a divides s, that's a violation.
    # So for each 3-sum s, we need to check if any a in S\{x,y,z} divides s.
    
    # This is also complex. Let's stick with the per-element check but optimize it.
    
    def is_valid(S):
        """S is a set of integers. Returns True if valid."""
        S_list = sorted(S)
        k = len(S_list)
        if k < 4:
            return True
        
        # Precompute all 3-sums from S
        # Actually, let's just check per element.
        for a in S_list:
            others = [x for x in S_list if x != a]
            # Check if any 3-sum from others is divisible by a
            # To speed up, we can break early.
            # Also, we only need to check multiples of a in the possible range.
            # The min sum from others is at least 1+2+3=6 (if 1,2,3 are in others)
            # The max sum is at most 190+191+192=573
            
            # Let's iterate through forbidden multiples and see if any can be formed.
            # But checking if a sum can be formed is O(n^2) per sum (meet in middle for 3-sums).
            # Actually, for a fixed target sum T, checking if 3 distinct elements from 'others' sum to T is O(n^2).
            # There are at most 573/a multiples. For a=1, that's 568 multiples. 568 * 47^2 ~ 1.2M. Too slow.
            
            # Let's just iterate all triples. C(47,3) = 16215.
            # For 48 elements, 48 * 16215 = 778,320 iterations.
            # Each iteration is a modulo check. This should be fast enough in Python if we're careful.
            
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True
    
    def count_violations(S):
        """Returns number of violations."""
        S_list = sorted(S)
        k = len(S_list)
        if k < 4:
            return 0
        
        count = 0
        for a in S_list:
            others = [x for x in S_list if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    count += 1
        return count
    
    def get_violation_elements(S):
        """Returns a set of elements involved in violations."""
        S_list = sorted(S)
        k = len(S_list)
        if k < 4:
            return set()
        
        involved = set()
        for a in S_list:
            others = [x for x in S_list if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    involved.add(a)
                    involved.add(b)
                    involved.add(c)
                    involved.add(d)
        return involved
    
    # Strategy: Min-conflicts local search
    # Start with a random 48-element subset.
    # While not valid and time remains:
    #   Find an element involved in a violation.
    #   Try to swap it with a random element not in S.
    #   If the new set has fewer violations (or is valid), accept.
    #   Otherwise, try a different element or accept with some probability (simulated annealing).
    
    # Let's also try a greedy construction as a starting point.
    
    def greedy_construct():
        """Greedy: add elements one by one, starting from large numbers."""
        S = set()
        # Try adding elements in descending order
        for x in range(n, 0, -1):
            if len(S) >= max_size:
                break
            S_trial = S | {x}
            if is_valid(S_trial):
                S = S_trial
        return S
    
    def local_search(S, max_iterations=10000):
        """Min-conflicts local search."""
        S = set(S)
        current_violations = count_violations(S)
        best_S = set(S)
        best_violations = current_violations
        
        universe = list(range(1, n + 1))
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            if current_violations == 0:
                break
            
            # Find elements involved in violations
            involved = get_violation_elements(S)
            if not involved:
                break
            
            # Pick a random element from involved
            x = random.choice(list(involved))
            
            # Try to swap x with a random element y not in S
            # Or just remove x and add a random y
            remaining = [y for y in universe if y not in S]
            if not remaining:
                break
            
            y = random.choice(remaining)
            
            S_trial = (S - {x}) | {y}
            new_violations = count_violations(S_trial)
            
            if new_violations < current_violations:
                S = S_trial
                current_violations = new_violations
                if current_violations < best_violations:
                    best_S = set(S)
                    best_violations = current_violations
            elif new_violations == current_violations:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    S = S_trial
                    current_violations = new_violations
            else:
                # Reject, but maybe accept with very small probability (annealing)
                # For simplicity, just reject
                pass
        
        return best_S, best_violations
    
    # First, try greedy
    S_greedy = greedy_construct()
    if len(S_greedy) == max_size and is_valid(S_greedy):
        return list(S_greedy)
    
    # Try local search from greedy start
    S_ls, v_ls = local_search(S_greedy, max_iterations=5000)
    if len(S_ls) == max_size and is_valid(S_ls):
        return list(S_ls)
    
    # Try multiple random starts
    best_S = S_greedy
    best_v = count_violations(S_greedy)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random 48-element subset
        S_random = set(random.sample(range(1, n + 1), max_size))
        v_random = count_violations(S_random)
        
        if v_random < best_v:
            best_S = S_random
            best_v = v_random
        
        # Local search from this start
        S_ls, v_ls = local_search(S_random, max_iterations=2000)
        if v_ls < best_v:
            best_S = S_ls
            best_v = v_ls
        
        if best_v == 0:
            break
    
    # If we found a valid set, return it
    if is_valid(best_S):
        return list(best_S)
    
    # If not, try to repair the best set found
    S = set(best_S)
    while not is_valid(S) and time.time() < deadline:
        involved = get_violation_elements(S)
        if not involved:
            break
        
        # Try removing an involved element
        x = random.choice(list(involved))
        S.remove(x)
        
        # Try to add back an element
        remaining = [y for y in range(1, n + 1) if y not in S]
        random.shuffle(remaining)
        for y in remaining:
            S_trial = S | {y}
            if is_valid(S_trial):
                S = S_trial
                break
    
    # Final attempt: try all pairs of swaps
    if not is_valid(S) and time.time() < deadline:
        S_list = list(S)
        remaining = [y for y in range(1, n + 1) if y not in S]
        for x in S_list:
            if time.time() > deadline:
                break
            for y in remaining:
                S_trial = (S - {x}) | {y}
                if is_valid(S_trial):
                    S = S_trial
                    remaining = [z for z in range(1, n + 1) if z not in S]
                    S_list = list(S)
                    break
    
    return list(S)
