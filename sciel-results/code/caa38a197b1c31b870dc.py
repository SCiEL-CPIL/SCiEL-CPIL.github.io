import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    def find_solution(k, max_den=10000000):
        """
        Find k distinct denominators in [5, max_den] summing to 1.
        Uses DFS with integer arithmetic.
        State: remaining_num, remaining_den, k_left, min_den, current
        """
        # remaining = num / den
        def dfs(num, den, k_left, min_den, current):
            if time.time() > deadline:
                return None
            
            if k_left == 0:
                if num == 0:
                    return current[:]
                return None
            
            if k_left == 1:
                # 1/x = num/den => x = den/num
                if num > 0 and den % num == 0:
                    x = den // num
                    if x > min_den and x <= max_den:
                        return current + [x]
                return None
            
            # Bounds for next denominator a
            # 1/a < num/den => a > den/num
            # k_left/a > num/den => a < k_left * den / num
            
            lower_a = max(min_den, den // num + 1)
            upper_a = min(max_den, (k_left * den) // num + 1)
            
            if lower_a > upper_a:
                return None
            
            for a in range(lower_a, upper_a + 1):
                if time.time() > deadline:
                    return None
                
                # Prune: max possible sum
                # sum_{i=0}^{k_left-1} 1/(a+i) < num/den
                # Approximate: k_left/a < num/den => a < k_left*den/num (already used)
                # Tighter: if even the max sum is too small, break
                # We can check if 1/a + 1/(a+1) + ... < num/den
                # For small k_left, just calculate.
                max_sum_num = 0
                max_sum_den = 1
                for i in range(k_left):
                    # max_sum += 1/(a+i)
                    # max_sum = max_sum + 1/(a+i)
                    # To avoid huge numbers, we can just check the first term or use float for pruning
                    pass # Float check is faster
                # Use float for pruning check
                max_possible = sum(1.0 / (a + i) for i in range(k_left))
                if max_possible < float(num) / float(den) - 1e-15:
                    break
                
                # new_remaining = num/den - 1/a = (num*a - den) / (den*a)
                new_num = num * a - den
                if new_num <= 0:
                    continue
                
                new_den = den * a
                # Simplify fraction
                g = gcd(new_num, new_den)
                new_num //= g
                new_den //= g
                
                result = dfs(new_num, new_den, k_left - 1, a + 1, current + [a])
                if result is not None:
                    return result
            
            return None
        
        return dfs(1, 1, k, 5, [])

    # Try k=8, 9, 10
    for k in range(8, 11):
        if time.time() > deadline - 5:
            break
        sol = find_solution(k, max_den=10000000)
        if sol is not None:
            # Verify
            s_num = 0
            s_den = 1
            for x in sol:
                # s += 1/x
                s_num = s_num * x + s_den
                s_den = s_den * x
                g = gcd(s_num, s_den)
                s_num //= g
                s_den //= g
            if s_num == s_den and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
                return sol

    # Fallback: Return a known valid solution or greedy
    # Greedy (standard)
    def greedy_solution():
        num, den = 1, 1
        denoms = []
        last = 4
        while num > 0:
            # Find smallest d > last such that 1/d <= num/den
            # d >= den/num
            d = max(last + 1, den // num + (1 if den % num != 0 else 0))
            # Ensure 1/d <= num/den
            while d * num > den:
                d += 1
            denoms.append(d)
            # new_num = num*d - den
            # new_den = den*d
            new_num = num * d - den
            new_den = den * d
            g = gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    # Verify greedy
    s_num = 0
    s_den = 1
    for x in sol:
        s_num = s_num * x + s_den
        s_den = s_den * x
        g = gcd(s_num, s_den)
        s_num //= g
        s_den //= g
    if s_num == s_den and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
