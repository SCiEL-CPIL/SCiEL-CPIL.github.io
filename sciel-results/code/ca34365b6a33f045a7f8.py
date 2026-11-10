import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    target = Fraction(5, 31)

    # Known good solution with max 248
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    best_solution = [8, 31, 248]
    best_max = 248

    # We will try to find a solution with a smaller maximum denominator.
    # The search space is defined by:
    # 1. The maximum denominator M.
    # 2. The number of terms K.
    # 3. The specific set of denominators.
    
    # Since we want to minimize M, we can iterate M downwards from 247.
    # For each M, we check if a valid Egyptian fraction exists with all d_i <= M.
    # If found, update best and continue.
    # If not found, we must be careful: it's possible the DFS pruned incorrectly or ran out of time.
    # However, for small M, the search space is manageable.
    
    # Let's optimize the DFS significantly.
    # Key optimization:
    # 1. Use integer arithmetic where possible to avoid Fraction overhead.
    #    We are solving sum(1/d_i) = N/D.
    #    Multiply by D * prod(d_i) is complex.
    #    Instead, maintain remaining fraction as (num, den).
    #    Start: num=5, den=31.
    #    If we pick d, new_num = num*d - den, new_den = den*d.
    #    We can simplify by gcd.
    
    # 2. Pruning:
    #    - If remaining num <= 0, fail.
    #    - If 1/d > remaining, skip (d too small).
    #    - If sum of remaining k-1 largest possible reciprocals < remaining, fail.
    #      The largest possible reciprocals are 1/(d_min+1), ..., 1/(d_min+k-1).
    #      Actually, we are picking distinct d > last_d.
    #      So the max possible sum for the remaining terms is sum_{i=1}^{k-1} 1/(last_d + i).
    #      If this sum < remaining, prune.
    #    - If sum of remaining k-1 smallest possible reciprocals > remaining, fail?
    #      The smallest possible reciprocals are 1/M, 1/(M-1), ...
    #      If this sum < remaining, we can't reach it? No, we need the sum to BE remaining.
    #      If the MAX possible sum < remaining, we fail.
    #      If the MIN possible sum > remaining, we fail.
    #      Min possible sum is sum of 1/M, 1/(M-1), ... for k-1 terms.

    def _dfs(num, den, last_d, max_d, k_left, used_set, path):
        if time.time() > deadline:
            return None
        if k_left == 0:
            if num == 0:
                return list(path)
            return None
        
        if num <= 0:
            return None
            
        # Lower bound for next d
        # 1/d < num/den  => d > den/num
        # So d_min = floor(den/num) + 1
        d_min = den // num + 1
        if d_min <= last_d:
            d_min = last_d + 1
            
        if d_min > max_d:
            return None
            
        # Pruning: Max possible sum of remaining k_left terms
        # We need to pick k_left distinct integers from [d_min, max_d].
        # The maximum sum is achieved by picking the smallest available integers: d_min, d_min+1, ...
        max_sum = Fraction(0, 1)
        d = d_min
        for _ in range(k_left):
            if d > max_d:
                break
            max_sum += Fraction(1, d)
            d += 1
        if max_sum < Fraction(num, den):
            return None
            
        # Pruning: Min possible sum of remaining k_left terms
        # The minimum sum is achieved by picking the largest available integers: max_d, max_d-1, ...
        min_sum = Fraction(0, 1)
        d = max_d
        for _ in range(k_left):
            if d < d_min:
                break
            min_sum += Fraction(1, d)
            d -= 1
        if min_sum > Fraction(num, den):
            return None

        # Iterate d from d_min to max_d
        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            if d in used_set:
                continue
                
            # New remaining fraction
            # num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                break # Since d is increasing, new_num will only decrease
            
            # Simplify
            g = __import__('math').gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            used_set.add(d)
            path.append(d)
            res = _dfs(new_num, new_den, d, max_d, k_left - 1, used_set, path)
            path.pop()
            used_set.discard(d)
            
            if res is not None:
                return res
                
        return None

    # Search for smaller max denominator
    # We try M from 247 down to 10.
    # For each M, we try different numbers of terms k.
    # What is the minimum number of terms?
    # 1/8 + 1/31 + 1/248 is 3 terms.
    # Can we do 2 terms? 1/a + 1/b = 5/31.
    # 31(a+b) = 5ab => 155a + 155b = 25ab => 25ab - 155a - 155b = 0
    # (25a - 155)(25b - 155) = 155^2 = 24025
    # Let X = 25a - 155, Y = 25b - 155. XY = 24025.
    # 24025 = 5^2 * 7^2 * 11^2? No.
    # 155 = 5 * 31. 155^2 = 25 * 961 = 25 * 31^2.
    # Factors of 24025:
    # 1, 5, 25, 31, 155, 775, 961, 4805, 24025.
    # Pairs (X, Y) such that X < Y (WLOG a < b).
    # X = 1 => Y = 24025 => 25a = 156 => a not integer.
    # X = 5 => Y = 4805 => 25a = 160 => a=6.4 No.
    # X = 25 => Y = 961 => 25a = 180 => a=7.2 No.
    # X = 31 => Y = 775 => 25a = 186 => a=7.44 No.
    # X = 155 => Y = 155 => 25a = 310 => a=12.4 No.
    # So no 2-term solution with integer denominators.
    
    # So we need at least 3 terms.
    # Let's try k=3 first for each M. If not found, try k=4, etc.
    # But increasing k increases search time.
    # We should prioritize small k.
    
    # Also, we can bound the minimum number of terms.
    # If sum(1/M + 1/(M-1) + ... + 1/(M-k+1)) < 5/31, then k is too small.
    
    # Let's iterate M downwards.
    # For efficiency, we can start with M=247 and go down.
    # But maybe a solution with M=100 exists with k=5, while M=200 has no solution?
    # Yes. So we must check all M.
    
    # To speed up, we can pre-calculate the minimum k required for a given M.
    # min_k(M) = smallest k such that sum_{i=0}^{k-1} 1/(M-i) >= 5/31.
    
    for M in range(best_max - 1, 10, -1):
        if time.time() > deadline - 2:
            break
            
        # Find min k
        s = 0.0
        k_min = 1
        for i in range(M, 0, -1):
            s += 1.0 / i
            if s >= 5.0 / 31.0:
                k_min = M - i + 1
                break
        else:
            k_min = M # Should not happen for small M
            
        # Try k from k_min to some reasonable max (e.g., 10)
        # We can cap k at 10 or so, as larger k is very slow.
        found = False
        for k in range(k_min, 11):
            if time.time() > deadline - 2:
                break
                
            res = _dfs(5, 31, 0, M, k, set(), [])
            if res is not None:
                # Verify
                if len(set(res)) == len(res) and all(1 <= d <= 10000000 for d in res):
                    s_frac = sum(Fraction(1, d) for d in res)
                    if s_frac == target:
                        max_d = max(res)
                        if max_d < best_max:
                            best_max = max_d
                            best_solution = res
                            # We found a better solution. We can continue searching for even smaller M.
                            # But we should update the loop bound?
                            # The loop is iterating M downwards.
                            # If we found a solution with max_d, we can break this inner loop and continue outer.
                            # Actually, we want the MINIMUM max denominator.
                            # So if we found one at M, we should check if there's one at M-1, M-2, etc.
                            # So we do NOT break the outer loop.
                            # We just update best_solution and continue.
                            found = True
                            break # Break inner k loop, try next M
                            
        # If no solution found for this M, continue to M-1.
        # Note: It is possible that a solution exists for M but our DFS failed due to time/pruning.
        # But for small M, it should be robust.

    return best_solution
