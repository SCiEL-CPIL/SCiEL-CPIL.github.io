<<<<<<< SEARCH
    # Try to find a 9-term solution using targeted DFS with integer arithmetic
    def dfs(rem_num, rem_den, min_d, terms_left, current):
        if time.time() > deadline:
            return None
        if terms_left == 0:
            return None if rem_num != 0 else []
        
        # Each term <= 1/min_d, so terms_left/min_d >= rem_num/rem_den
        if terms_left * rem_den < rem_num * min_d:
            return None
        
        # d must satisfy: 1/d <= rem_num/rem_den => d >= ceil(rem_den/rem_num)
        d_lower = max(min_d, -(-rem_den // rem_num))
        
        # Upper bound for d: we need the remaining fraction to be positive
        # rem_num/rem_den - 1/d > 0 => d > rem_den/rem_num (already have)
        # Also need (terms_left-1) terms to sum to rem_num/rem_den - 1/d
        # Max sum of (terms_left-1) terms starting from d+1 is (terms_left-1)/(d+1)
        # So: rem_num/rem_den - 1/d <= (terms_left-1)/(d+1)
        # This gives an upper bound; use a loose one
        d_upper = min(10000000, rem_den * terms_left + 1000)
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            if rem_num * d <= rem_den:
                continue
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Prune: max possible sum of (terms_left-1) terms starting at d+1
            if (terms_left - 1) * (d + 1) < new_num * new_den:
                break
            
            # Prune: min possible sum of (terms_left-1) terms is (terms_left-1)/10000000
            if (terms_left - 1) * new_den < new_num * 10000000:
                continue
            
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return current + [d] + result
        
        return None

    # Try 9 terms with time limit
    if time.time() < deadline - 10:
        result = dfs(1, 1, 5, 9, [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
=======
    # Try to find a solution using targeted DFS with integer arithmetic
    def dfs(rem_num, rem_den, min_d, terms_left, current):
        if time.time() > deadline:
            return None
        if terms_left == 0:
            return None if rem_num != 0 else []
        
        # Each term <= 1/min_d, so terms_left/min_d >= rem_num/rem_den
        if terms_left * rem_den < rem_num * min_d:
            return None
        
        # d must satisfy: 1/d <= rem_num/rem_den => d >= ceil(rem_den/rem_num)
        d_lower = max(min_d, -(-rem_den // rem_num))
        
        # Tight upper bound: 1/d >= rem_num / (terms_left * rem_den)
        # d <= terms_left * rem_den / rem_num
        d_upper = (terms_left * rem_den) // rem_num
        d_upper = min(d_upper, 10000000)
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            if rem_num * d <= rem_den:
                continue
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Prune: max possible sum of (terms_left-1) terms starting at d+1
            if (terms_left - 1) * (d + 1) < new_num * new_den:
                break
            
            # Prune: min possible sum of (terms_left-1) terms is (terms_left-1)/10000000
            if (terms_left - 1) * new_den < new_num * 10000000:
                continue
            
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return current + [d] + result
        
        return None

    # Try 8, 9, 10, 11 terms with time limit
    for k in range(8, 12):
        if time.time() > deadline - 5:
            break
        result = dfs(1, 1, 5, k, [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
>>>>>>> REPLACE

