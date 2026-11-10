import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # k=5,6,7,8 are impossible as sum of reciprocals of first k integers >= 5 is < 1.
    # k=9 is the first possible.
    for k in range(9, 257):
        if time.time() > deadline:
            break
            
        def dfs(depth, prev_den, rem):
            if time.time() > deadline:
                return None
            
            if depth == k:
                if rem == 0:
                    return []
                return None
            
            m = k - depth
            # rem is a Fraction
            num, den = rem.numerator, rem.denominator
            
            # Lower bound for next denominator
            # 1/d <= rem => d >= 1/rem = den/num
            min_d = max(prev_den + 1, (den + num - 1) // num)
            
            # Upper bound for next denominator
            # rem <= m/d => d <= m/rem = m * num / den
            max_d = (m * num) // den
            
            if min_d > max_d:
                return None
            
            # Iterate possible denominators
            for d in range(min_d, max_d + 1):
                # min_d ensures 1/d <= rem, so new_rem >= 0
                new_rem = rem - Fraction(1, d)
                
                # Pruning: Check if remaining sum can be formed by m-1 terms
                # Max sum of m-1 terms with denom > d is (m-1)/(d+1)
                # If new_rem > (m-1)/(d+1), it's impossible.
                if m > 1:
                    if new_rem > Fraction(m - 1, d + 1):
                        continue
                        
                res = dfs(depth + 1, d, new_rem)
                if res is not None:
                    return [d] + res
            
            return None

        result = dfs(0, 4, Fraction(1, 1))
        if result is not None:
            return result
            
    # Fallback to the original DP if DFS fails or times out
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
