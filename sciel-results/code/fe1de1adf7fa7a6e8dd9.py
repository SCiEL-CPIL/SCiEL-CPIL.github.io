import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute: for each k, the max possible sum with k terms (denominators 5..4+k)
    # k terms: 1/5 + 1/6 + ... + 1/(4+k)
    # Find smallest k where this sum >= 1
    min_k = 5
    s = 0.0
    for k in range(5, 100):
        s += 1.0 / (4 + k)
        if s >= 1.0:
            min_k = k
            break
    
    for k in range(min_k, 257):
        if time.time() > deadline:
            break
            
        def dfs(depth, prev_den, rem_num, rem_den):
            if time.time() > deadline:
                return None
            
            if depth == k:
                if rem_num == 0:
                    return []
                return None
            
            m = k - depth
            
            # rem is rem_num / rem_den
            # Lower bound for next denominator
            # 1/d <= rem => d >= 1/rem = rem_den/rem_num
            min_d = max(prev_den + 1, (rem_den + rem_num - 1) // rem_num)
            
            # Upper bound for next denominator
            # rem <= m/d => d <= m/rem = m * rem_num / rem_den
            max_d = (m * rem_num) // rem_den
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                # new_rem = rem_num/rem_den - 1/d
                # new_num = rem_num*d - rem_den
                # new_den = rem_den*d
                new_num = rem_num * d - rem_den
                new_den = rem_den * d
                
                if new_num < 0:
                    continue
                    
                # Simplify new_rem
                g = math.gcd(new_num, new_den)
                new_num //= g
                new_den //= g
                
                # Pruning: Check if remaining sum can be formed by m-1 terms
                # Max sum of m-1 terms with denom > d is (m-1)/(d+1)
                # If new_rem > (m-1)/(d+1), it's impossible.
                if m > 1:
                    # new_num/new_den > (m-1)/(d+1)
                    # new_num * (d+1) > new_den * (m-1)
                    if new_num * (d + 1) > new_den * (m - 1):
                        continue
                    
                    # Also check min possible sum: (m-1)/max_den where max_den is unbounded
                    # But more useful: if new_rem < 1/max_d_remaining, skip
                    # Actually, the lower bound on next denominators is d+1, so
                    # min sum of m-1 terms is 1/(d+1) + 1/(d+2) + ... 
                    # This is hard to bound tightly, skip for now
                
                res = dfs(depth + 1, d, new_num, new_den)
                if res is not None:
                    return [d] + res
            
            return None

        result = dfs(0, 4, 1, 1)
        if result is not None:
            return result
            
    # Fallback: DP over divisors of a common denominator
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
