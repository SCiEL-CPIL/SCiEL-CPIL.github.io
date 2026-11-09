import time
import sys
from math import gcd
from fractions import Fraction

def construct():
    """
    Returns 1..256 distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1. Minimizes their count.
    """
    start_time = time.time()
    timeout = 58.0  # Safety margin before 60s kill
    
    def check_time():
        return time.time() - start_time > timeout

    # Strategy: 
    # The standard greedy algorithm for Egyptian fractions produces a valid solution,
    # but it is not minimal in count. 
    # However, we can use a search to find a solution with a smaller count.
    # The minimum number of terms for 1 with denominators >= 5 is likely small.
    # Let's try to find a solution with k terms, starting from k=5 or k=6.
    # Actually, it is known that 1 = 1/2 + 1/3 + 1/6. 
    # With min denom 5, we can split 1/2 = 1/3 + 1/6 -> 1/4+1/12, etc.
    # Let's just do a DFS search for the minimum k.
    
    def search_k(k, max_d=10000000):
        if check_time():
            return None
        # Remaining sum: 1
        # We need to pick k distinct denominators d_1 < d_2 < ... < d_k such that sum(1/d_i) = 1.
        # d_1 >= 5.
        
        def dfs(remaining, next_min_d, terms_left, chosen):
            if check_time():
                return "TIMEOUT"
            
            if terms_left == 0:
                if remaining == 0:
                    return chosen[:]
                return None
            
            # Lower bound for next denominator:
            # 1/d <= remaining => d >= 1/remaining
            # Also d >= next_min_d
            # Also, we must be able to form 'remaining' with 'terms_left' terms starting from 'next_min_d'.
            # Max sum possible with terms_left terms starting from next_min_d is approx terms_left/next_min_d
            # If remaining > terms_left/next_min_d, it's impossible? No, that's a lower bound on the sum of reciprocals?
            # Actually, the sum of reciprocals decreases as denominators increase.
            # The maximum sum is obtained by taking the smallest possible denominators.
            # So, max_sum = sum(1/i for i in range(next_min_d, next_min_d + terms_left))
            # If remaining > max_sum, prune.
            
            # Min sum is obtained by taking the largest possible denominators?
            # But denominators are bounded by max_d.
            # However, for a lower bound on d, we know:
            # remaining = 1/d_1 + ... 
            # 1/d_1 < remaining => d_1 > 1/remaining
            # So d_1 >= floor(1/remaining) + 1.
            
            min_d = max(next_min_d, int(1 / remaining) + 1)
            
            # Also, we need to leave enough sum for the remaining terms.
            # The largest possible sum for the remaining 'terms_left - 1' terms 
            # starting from d+1 is approximately (terms_left - 1) / (d+1).
            # So, 1/d + (terms_left - 1)/(d+1) >= remaining.
            # This gives an upper bound for d.
            # 1/d + (terms_left - 1)/(d+1) >= remaining
            # This is an inequality to find max_d.
            # For small d, this is easier.
            # Let's iterate d from min_d upwards, but with an upper bound.
            
            # A simple upper bound: 
            # If we pick d too large, the sum of reciprocals will be too small to reach 'remaining'.
            # The sum of the remaining 'terms_left - 1' terms is at most (terms_left - 1) / (d + 1).
            # So 1/d + (terms_left - 1)/(d + 1) >= remaining.
            # If this fails, we can stop.
            
            # Let's refine min_d.
            # We need 1/d <= remaining, so d >= 1/remaining.
            # Actually, 1/d can be equal to remaining if terms_left == 1.
            
            if terms_left == 1:
                if remaining.denominator == 1:
                    d = remaining.numerator
                    if d >= min_d and d <= max_d and (not chosen or d > chosen[-1]):
                        return chosen + [d]
                return None
            
            # Iterate d
            # Upper bound for d:
            # 1/d + (terms_left - 1)/(d + 1) >= remaining
            # This implies d cannot be too large.
            # Approx: terms_left / d >= remaining => d <= terms_left / remaining.
            
            max_d_bound = min(max_d, int(terms_left / remaining))
            
            # Also, we must be able to form the remaining sum.
            # Let's just iterate and prune.
            
            for d in range(min_d, max_d_bound + 1):
                if d in chosen:
                    continue
                
                next_remaining = remaining - Fraction(1, d)
                if next_remaining < 0:
                    break  # d is too small? No, d increases, 1/d decreases.
                    # If next_remaining < 0, it means 1/d > remaining.
                    # As d increases, 1/d decreases.
                    # So if next_remaining < 0, we should continue? 
                    # No, min_d is calculated as floor(1/remaining) + 1, so 1/d < remaining for all d >= min_d?
                    # Wait. If remaining = 0.2, 1/remaining = 5. min_d = 6. 1/6 < 0.2.
                    # So next_remaining > 0.
                    # So this break should not trigger if min_d is correct.
                
                # Prune: Can we form next_remaining with terms_left-1 terms starting from d+1?
                # Max sum is approx (terms_left - 1) / (d + 1).
                # If next_remaining > (terms_left - 1) / (d + 1), prune.
                if next_remaining > Fraction(terms_left - 1, d + 1):
                    continue
                
                # Also, min sum?
                # Min sum is with largest denominators. But we don't know the max denominator yet.
                # However, if next_remaining is too small to be formed by terms_left-1 terms 
                # even with very large denominators? No, we can always make the sum small.
                # The constraint is that we need *distinct* denominators.
                
                res = dfs(next_remaining, d + 1, terms_left - 1, chosen + [d])
                if res is not None:
                    return res
            
            return None
        
        # Start search for k=5, 6, etc.
        # It is known that 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ... 
        # Actually, the greedy algorithm gives a solution.
        # But we want the minimum count.
        # Let's try k from 5 to 20.
        
        for k in range(5, 21):
            if check_time():
                break
            res = dfs(Fraction(1, 1), 5, k, [])
            if res is not None and res != "TIMEOUT":
                # Verify
                if len(set(res)) == len(res) and all(5 <= x <= 10000000 for x in res):
                    if sum(Fraction(1, x) for x in res) == 1:
                        return res
        
        # If search fails, use greedy algorithm to ensure a valid solution is returned.
        # Greedy algorithm:
        # remaining = 1
        # chosen = []
        # while remaining > 0:
        #   d = ceil(1/remaining)
        #   if d < 5: d = 5
        #   if d in chosen: d += 1
        #   chosen.append(d)
        #   remaining -= 1/d
        #   if remaining == 0: break
        #   if len(chosen) > 256: raise
        # This will produce a valid solution, though not necessarily minimal.
        
        if check_time():
            # Return a hardcoded small solution if we have one?
            # Or just return an empty list? No, must be valid.
            # Let's try to compute a greedy solution quickly.
            pass
            
        # Fallback: Greedy
        remaining = Fraction(1, 1)
        chosen = []
        d = 5
        while remaining > 0:
            if len(chosen) >= 256:
                break
            # Find smallest d >= 5 not in chosen such that 1/d <= remaining
            # Actually, greedy picks d = ceil(1/remaining).
            # If d < 5, set d = 5.
            # If d in chosen, increment d until not in chosen.
            
            next_d = int(1 / remaining) + 1
            if next_d < 5:
                next_d = 5
            
            while next_d in chosen:
                next_d += 1
            
            # Check if 1/next_d <= remaining
            # If not, we need a larger next_d? No, next_d is ceil(1/remaining) so 1/next_d <= remaining.
            # Wait, if remaining = 1/3, ceil(3) = 3. 1/3 <= 1/3. OK.
            # If remaining = 1/4 + 1/5, 1/remaining is not integer.
            
            chosen.append(next_d)
            remaining -= Fraction(1, next_d)
            
            if check_time():
                # This might take a while if the solution is long, but usually it's short.
                # If we time out, we return what we have? No, must be exact sum 1.
                # So we cannot return a partial sum.
                # But greedy is very fast.
                pass
        
        if sum(Fraction(1, x) for x in chosen) == 1:
            return chosen
            
        # Last resort: Hardcoded solution?
        # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
        # This is not exact.
        
        # Let's use a known valid solution:
        # 1 = 1/2 + 1/3 + 1/6
        # Split 1/2 -> 1/3 + 1/6. But 1/3 is already used? No, denominators must be distinct.
        # 1/2 = 1/4 + 1/4. No, distinct.
        # 1/2 = 1/5 + 1/20 + 1/20? No.
        # 1/2 = 1/5 + 1/10 + 1/20? 1/5+1/10+1/20 = 4/20+2/20+1/20 = 7/20 != 10/20.
        # 1/2 = 1/5 + 1/10 + 1/20 + 1/40 + ...
        
        # Actually, the greedy algorithm is robust. Let's just rely on it.
        # If it fails, raise an error.
        
        raise RuntimeError("No solution found")

    return search_k(5)

# To ensure the function is defined correctly and doesn't crash on import
if __name__ == "__main__":
    res = construct()
    print(res)
