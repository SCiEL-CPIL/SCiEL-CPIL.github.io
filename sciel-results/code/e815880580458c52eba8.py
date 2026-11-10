from fractions import Fraction

def construct():
    max_denom = 10**7
    
    def solve(k_target):
        def max_sum(start, k):
            s = Fraction(0, 1)
            for i in range(k):
                s += Fraction(1, start + i)
            return s

        def dfs(start, rem, current):
            k_rem = k_target - len(current)
            if k_rem == 0:
                if rem == 0:
                    return current[:]
                return None
            
            # If rem is 0, we need to add k_rem terms summing to 0. Impossible.
            if rem == 0:
                return None

            # Upper bound for d
            d_max = int((k_rem * rem.denominator) // rem.numerator)
            if d_max > max_denom:
                d_max = max_denom
            
            if max_sum(start, k_rem) < rem:
                return None

            for d in range(start, d_max + 1):
                if d > max_denom:
                    break
                
                if max_sum(d, k_rem) < rem:
                    break
                
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                
                # If new_rem is 0 but we still need more terms, it's impossible
                if new_rem == 0 and k_rem > 1:
                    continue

                res = dfs(d + 1, new_rem, current + [d])
                if res is not None:
                    return res
            
            return None

        return dfs(5, Fraction(1, 1), [])

    # Try k=8
    res = solve(8)
    if res:
        return res
        
    # Try k=9
    res = solve(9)
    if res:
        return res

    # Try k=10
    res = solve(10)
    if res:
        return res

    # Fallback
    # This should not happen for k=8
    return []
