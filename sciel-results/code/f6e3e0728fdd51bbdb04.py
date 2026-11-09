import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    def search(k, deadline):
        # rem is Fraction
        # last_d is int
        # depth is int (number of terms chosen so far)
        # path is list of denominators
        # max_depth is k
        def dfs(rem, last_d, depth, path):
            if time.time() > deadline:
                return None
            
            if depth == k:
                if rem == 0:
                    return path
                return None
            
            # Pruning: Max possible sum of remaining (k - depth) terms
            # with denominators > last_d
            # The max sum is achieved by smallest denominators: last_d+1, ..., last_d + (k-depth)
            remaining_terms = k - depth
            max_sum = Fraction(0, 1)
            for i in range(remaining_terms):
                max_sum += Fraction(1, last_d + 1 + i)
            
            if rem > max_sum:
                return None
            
            # Lower bound for next denominator d
            # 1/d <= rem => d >= 1/rem
            # Also d > last_d
            if rem <= 0:
                return None
            
            min_d = last_d + 1
            # d must be at least ceil(1/rem)
            # 1/rem is a Fraction. d >= 1/rem => d * rem >= 1
            # d >= 1/rem
            # Let's compute ceil(1/rem)
            # 1/rem = 1 / (num/den) = den/num
            # ceil(den/num)
            num, den = rem.numerator, rem.denominator
            # We need d * num >= den => d >= den / num
            ceil_val = (den + num - 1) // num
            if ceil_val > min_d:
                min_d = ceil_val
            
            # Upper bound for d?
            # We need rem - 1/d to be achievable by remaining_terms - 1 terms.
            # Min sum of remaining terms (excluding 1/d) is 0 (approx).
            # But we also need rem - 1/d >= 0 => 1/d <= rem => d >= 1/rem (already handled).
            # Also, the remaining sum rem - 1/d must be <= max sum of remaining terms - 1.
            # This is implicitly handled by the recursive call's pruning.
            
            # We can also limit d to avoid huge numbers if needed, but 10^7 is fine.
            # However, if d is very large, 1/d is small.
            # The loop will break if 1/d < rem - max_possible_remaining_sum?
            # No, if 1/d is too small, rem - 1/d is large, and might exceed max sum of remaining.
            
            for d in range(min_d, 10000001):
                if time.time() > deadline:
                    return None
                
                term = Fraction(1, d)
                if term > rem:
                    # Since d increases, term decreases.
                    # If term > rem, we need larger d? No, 1/d decreases as d increases.
                    # So if 1/d > rem, then for smaller d it was even larger.
                    # We started from min_d which ensures 1/d <= rem.
                    # So this condition should not trigger if min_d is correct.
                    # But due to integer arithmetic in ceil, it might be off by 1.
                    # If term > rem, we can't use this d. We need larger d?
                    # No, if 1/d > rem, then d < 1/rem.
                    # Our min_d is ceil(1/rem). So d >= 1/rem => 1/d <= rem.
                    # So term should always be <= rem.
                    pass
                
                new_rem = rem - term
                if new_rem < 0:
                    continue
                
                # Pruning check for new state
                # Max sum of remaining_terms - 1 terms with denominators > d
                rem_terms_left = remaining_terms - 1
                if rem_terms_left > 0:
                    max_next_sum = Fraction(0, 1)
                    for i in range(rem_terms_left):
                        max_next_sum += Fraction(1, d + 1 + i)
                    if new_rem > max_next_sum:
                        # If new_rem is too large, we need smaller d?
                        # No, smaller d means larger term, smaller new_rem.
                        # We are iterating d upwards.
                        # If new_rem > max_next_sum, it means we subtracted too little?
                        # No, 1/d is small (d large), so new_rem is large.
                        # We need new_rem to be small enough to be covered by remaining terms.
                        # So we need larger term (smaller d).
                        # But we are iterating d upwards.
                        # So if this condition holds, it will hold for all larger d.
                        # So we can break.
                        break
                else:
                    if new_rem != 0:
                        continue
                
                path.append(d)
                result = dfs(new_rem, d, depth + 1, path)
                if result:
                    return result
                path.pop()
            
            return None

        result = dfs(Fraction(1, 1), 4, 0, [])
        return result

    # Try k=8
    res = search(8, deadline)
    if res:
        return res

    # Try k=9
    res = search(9, deadline)
    if res:
        return res

    # Fallback: Hardcoded solution or seed-like
    # The seed program finds a solution. We can just return a known valid one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + ...
    # This is too long.
    # Let's use the divisors of 2520 logic from the seed as a fallback, but ensure it's distinct and >= 5.
    # The seed's output is valid.
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
    
    raise RuntimeError("no solution found")
