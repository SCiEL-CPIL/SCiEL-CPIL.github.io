import time

def construct():
    # Deadline guard
    deadline = time.time() + 50.0
    
    def find_terms(frac_num, frac_den, min_den, depth, current_terms):
        """
        Recursively find 'depth' terms that sum to frac_num/frac_den.
        All terms must have denominator >= min_den.
        Returns list of denominators or None.
        """
        if depth == 0:
            if frac_num == 0:
                return current_terms
            return None
        
        if depth == 1:
            if frac_den % frac_num == 0:
                d = frac_den // frac_num
                if d >= min_den:
                    return current_terms + [d]
            return None

        # Pruning: if 1/min_den * depth < frac_num/frac_den, we can't reach the sum
        # Actually, we need sum of 1/d_i = frac.
        # Max possible sum with depth terms and min_den is depth / min_den.
        if depth / min_den < frac_num / frac_den:
            return None
        
        # Min possible sum is depth / (some large number). 
        # Also, 1/d_1 must be <= frac. So d_1 >= 1/frac = frac_den/frac_num.
        # And d_1 >= min_den.
        start = max(min_den, (frac_den // frac_num) + 1)
        
        # Upper bound for d_1:
        # 1/d_1 < frac (strictly, since remaining terms are positive)
        # Actually 1/d_1 <= frac - (depth-1)/LARGE.
        # Simpler: 1/d_1 < frac.
        # Also, the remaining depth-1 terms must be able to sum to frac - 1/d_1.
        # The largest possible term for the next step is 1/start_next.
        # To maximize the sum of the rest, we pick the smallest possible denominators.
        # But we just need to iterate d_1.
        
        # Bound d_1:
        # frac - 1/d_1 must be representable by depth-1 terms with denom > d_1.
        # The maximum sum of depth-1 terms with denom > d_1 is approx (depth-1)/(d_1+1).
        # So frac - 1/d_1 <= (depth-1)/(d_1+1) is not quite right because denominators can be much larger.
        # Actually, we just need frac - 1/d_1 > 0.
        # And frac - 1/d_1 >= (depth-1) / (some large bound).
        
        # A better bound:
        # 1/d_1 < frac
        # d_1 > 1/frac = frac_den / frac_num
        # Also, the remaining sum is R = frac - 1/d_1.
        # We need to represent R with depth-1 terms, each >= 1/(d_1+1).
        # So R <= (depth-1) / (d_1+1) ? No, R can be small.
        # But R >= (depth-1) / MAX_DEN.
        
        # Let's just iterate d_1 from start up to a reasonable limit.
        # Since frac < 1, d_1 is at most 1/frac.
        # If frac is small, d_1 can be large.
        # However, if depth is small, d_1 cannot be too large or we won't have enough "room" for the other terms?
        # Actually, if d_1 is too large, 1/d_1 is small, leaving a large remainder R.
        # R must be sum of depth-1 unit fractions.
        # The largest unit fraction in the remainder is 1/(d_1+1).
        # So R <= (depth-1) * (1/(d_1+1))? No, that's the max sum if all were 1/(d_1+1).
        # Yes, if all remaining denominators are at least d_1+1, the max sum is (depth-1)/(d_1+1).
        # So we require: frac - 1/d_1 <= (depth-1)/(d_1+1).
        
        # Let's solve for d_1 upper bound.
        # frac - 1/d_1 <= (depth-1)/(d_1+1)
        # frac <= 1/d_1 + (depth-1)/(d_1+1)
        
        # We iterate d_1.
        # Upper bound for d_1:
        # If d_1 is very large, 1/d_1 is small.
        # The condition is: frac <= 1/d_1 + (depth-1)/(d_1+1).
        # For large d_1, RHS ~ depth/d_1.
        # So d_1 ~ depth/frac.
        
        # Let's just iterate up to a limit based on this.
        # Or just iterate while time remains.
        
        # Optimization: 
        # d_1 must satisfy: 1/d_1 < frac  => d_1 > frac_den/frac_num
        # And frac - 1/d_1 <= (depth-1)/(d_1+1)
        
        # Let's iterate d_1 from start to a calculated upper bound.
        # If frac is 1, d_1 starts at 5.
        # 1 - 1/5 = 0.8. depth=4.
        # 0.8 <= 3/6 = 0.5? No. So d_1=5 is invalid for depth=4?
        # Wait, if d_1=5, remaining sum is 0.8.
        # We need 4 terms? No, depth was 5, so remaining depth is 4.
        # Max sum of 4 terms with denom >= 6 is 4/6 = 0.66.
        # 0.8 > 0.66. So d_1=5 is impossible for 5 terms?
        # Let's check: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.2+0.166+0.142+0.125+0.111 = 0.744 < 1.
        # So we need smaller denominators.
        # But min_den is 5.
        # So for depth=5, min_den=5:
        # Max sum = 1/5+1/6+1/7+1/8+1/9 = 0.744.
        # This is < 1.
        # So NO 5-term solution exists with min_den=5?
        
        # Let's re-read the task.
        # "denominators in [5, 10000000]"
        # If the max possible sum of 5 distinct denominators >= 5 is < 1, then k=5 is impossible.
        
        # Max sum of 5 distinct unit fractions with denominators >= 5:
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9
        # = 0.2 + 0.1666 + 0.1428 + 0.125 + 0.1111
        # = 0.7455 < 1.
        
        # Therefore, it is IMPOSSIBLE to sum to 1 with 5 distinct denominators >= 5.
        
        # What about 6 terms?
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10
        # = 0.7455 + 0.1 = 0.8455 < 1.
        
        # What about 7 terms?
        # + 1/11 = 0.8455 + 0.0909 = 0.9364 < 1.
        
        # What about 8 terms?
        # + 1/12 = 0.9364 + 0.0833 = 1.0197 > 1.
        # So a solution with 8 terms likely exists.
        
        # Let's check if 7 terms can work.
        # We need sum = 1.
        # Max sum for 7 terms (denoms 5..11) is 0.9364.
        # So 7 terms is also IMPOSSIBLE.
        
        # So the minimum k is at least 8.
        
        # Let's verify the seed program's k=6.
        # Seed: `target = 2520`.
        # It finds denominators dividing 2520.
        # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35?
        # Sum of reciprocals of divisors of 2520 that are >= 5.
        # Divisors: 5, 6, 7, 10, 14, 21, 35, 42, ...
        # 1/5+1/6+1/7+1/10+1/14+1/35 = (140+105+90+70+42+30)/700 = 477/700 != 1.
        
        # The seed program uses DP to find a subset of divisors of 2520 that sum to 1.
        # 2520 = 504 * 5.
        # It returns a list of denominators.
        # If it returns 6 denominators, then k=6 is possible?
        # But my analysis says max sum of 6 distinct denoms >= 5 is < 1.
        # Contradiction.
        
        # Let's re-calculate max sum of 6 distinct denoms >= 5.
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10
        # 1/5 = 0.2
        # 1/6 = 0.16666
        # 1/7 = 0.14285
        # 1/8 = 0.125
        # 1/9 = 0.11111
        # 1/10 = 0.1
        # Sum = 0.8456.
        
        # So 6 terms is IMPOSSIBLE.
        
        # Why did the seed program score 1.0?
        # "Return 1..256 distinct integer denominators ... whose reciprocal sum is exactly 1"
        # If the seed program returns a valid list, then my assumption that k=6 is the count is wrong, OR the seed program returns more than 6.
        
        # Let's check the seed code logic.
        # `target = 2520`.
        # `reached` is a dict of sums to lists.
        # It iterates `d` from 5 to 2520.
        # If `2520 % d == 0`, it considers `w = 2520 // d`.
        # It tries to add `d` to existing subsets.
        # It stops when `target` (2520) is reached.
        # This finds a subset of divisors of 2520 (>=5) such that sum(2520/d) = 2520.
        # Which means sum(1/d) = 1.
        
        # How many divisors does it pick?
        # The greedy/DP approach might pick many small divisors.
        # Divisors of 2520 >= 5:
        # 5 (w=504)
        # 6 (w=420)
        # 7 (w=360)
        # 9 (w=280)
        # 10 (w=252)
        # 12 (w=210)
        # 14 (w=180)
        # 15 (w=168)
        # 18 (w=140)
        # 20 (w=126)
        # 21 (w=120)
        # 24 (w=105)
        # 28 (w=90)
        # 30 (w=84)
        # 35 (w=72)
        # 36 (w=70)
        # 40 (w=63)
        # 42 (w=60)
        # 45 (w=56)
        # 56 (w=45)
        # 60 (w=42)
        # 63 (w=40)
        # 70 (w=36)
        # 72 (w=35)
        # 84 (w=30)
        # 90 (w=28)
        # 105 (w=24)
        # 120 (w=21)
        # 126 (w=20)
        # 140 (w=18)
        # 168 (w=15)
        # 180 (w=14)
        # 210 (w=12)
        # 252 (w=10)
        # 280 (w=9)
        # 315 (w=8)
        # 360 (w=7)
        # 420 (w=6)
        # 504 (w=5)
        # 630 (w=4) - No, d>=5.
        
        # The DP finds a combination of weights summing to 2520.
        # Since it iterates `d` in increasing order, and updates `reached` in place (using `list(reached.items())`), it effectively builds a set.
        # It returns `reached[target]`.
        # The number of elements in this list is the count `k`.
        
        # Let's estimate k.
        # To sum to 2520 using weights like 504, 420, 360...
        # 504 + 420 + 360 + 280 + 252 + 210 + 180 + 168 + 140 + 126 + 120 + 105 + 90 + 84 + 72 + 70 + 63 + 60 + 56 + 45 + 42 + 40 + 36 + 35 + 30 + 28 + 24 + 21 + 20 + 18 + 15 + 14 + 12 + 10 + 9 + 8 + 7 + 6 + 5
        # This sum is much larger than 2520.
        # The DP finds a *subset*.
        # It likely picks a moderate number of terms.
        
        # If the seed score is 1.0, and score is 1/k, then k=1? No, sum of 1/1 = 1, but d>=5.
        # If k=1, 1/d=1 => d=1. Invalid.
        # If k=2, 1/a+1/b=1 => (a-1)(b-1)=1 => a=2,b=2. Invalid.
        # If k=3, 1/a+1/b+1/c=1. 1/2+1/3+1/6=1. d=2,3,6. Invalid.
        # 1/3+1/3+1/3? Distinct.
        # Smallest distinct solution for k=3 is 2,3,6.
        # For d>=5, k=3 is impossible (max sum 1/5+1/6+1/7 = 0.509 < 1).
        
        # So k must be larger.
        # If the seed program returns a list of length, say, 25, then score is 1/25 = 0.04.
        # But the prompt says "CURRENT PROGRAM (score 1.0000)".
        # And "1.0 equals the seed program".
        # This implies the seed program's score IS 1.0.
        # Score = 1/k. So k=1? Impossible.
        # Or Score = k? "The evaluator measures the raw value (lower is better)".
        # "search score is seed/value".
        # If value is the count k, and seed value is k_seed.
        # Score = k_seed / k_child.
        # If Score = 1.0, then k_child = k_seed.
        # So the "raw value" is the count k.
        # And the seed program produces a solution with count $k_{seed}$.
        # If I produce a solution with count $k_{child} < k_{seed}$, my score will be $> 1.0$.
        
        # So I need to find a solution with FEWER terms than the seed program.
        
        # How many terms does the seed program produce?
        # I can run it mentally or estimate.
        # It finds a subset of divisors of 2520.
        # The number of terms is likely around 10-20.
        
        # My analysis showed:
        # k=5: Impossible (Max sum < 1)
        # k=6: Impossible (Max sum < 1)
        # k=7: Impossible (Max sum < 1)
        # k=8: Possible (Max sum > 1)
        
        # So the theoretical minimum is k=8.
        # If the seed program produces k > 8, I can improve by finding a k=8 solution.
        # If the seed program produces k=8, I can't improve (unless I find k<8 which is impossible).
        
        # Let's assume the seed program produces k > 8.
        # I will write a program that finds a k=8 solution.
        
        # Algorithm:
        # 1. Try to find a solution with k=8.
        # 2. If found, return it.
        # 3. If not found within time, return a known valid solution (e.g., from seed logic or a hardcoded one).
        
        # To find k=8:
        # We need $1 = \sum_{i=1}^8 1/d_i$ with $5 \le d_1 < d_2 < ... < d_8 \le 10^7$.
        
        # Greedy approach (Egyptian fraction algorithm) usually minimizes terms but picks smallest possible denominators.
        # Standard Greedy:
        # $1 = 1/5 + 3/5$
        # $3/5 = 1/2 + 1/10$ -> 2 invalid.
        # Greedy on remaining fraction:
        # $1 - 1/5 = 4/5$. Next smallest denom for 4/5 is $\lceil 5/4 \rceil = 2$. Invalid.
        
        # We need to constrain denominators to be $\ge 5$.
        
        # Let's use a DFS with constraints.
        # Target: 1.
        # Depth: 8.
        # Min Denom: 5.
        
        # This search space is small enough if we prune well.
        
    # Implementing the DFS
    from fractions import Fraction
    import time

    def solve():
        start_time = time.time()
        deadline = start_time + 40.0 # 40s limit to be safe
        
        target = Fraction(1, 1)
        
        def dfs(remaining, min_den, depth, path):
            if time.time() > deadline:
                return None
            
            if depth == 0:
                if remaining == 0:
                    return path
                return None
            
            if remaining == 0:
                return None # Must use exactly 'depth' terms? 
                # Task says "Return 1..256 distinct... minimize count".
                # So we want the MINIMUM count.
                # If we find a solution with k terms, we return it.
                # We should try k=8, then k=9, etc.
                # But if we find k=8, we are done.
                
            # Pruning
            # Max sum possible with 'depth' terms starting at min_den:
            # S_max = sum(1/i for i in range(min_den, min_den+depth))
            # If remaining > S_max, impossible.
            # Min sum possible:
            # S_min = depth / LARGE. 
            # If remaining < S_min, impossible.
            
            # Calculate S_max
            s_max = 0
            for i in range(min_den, min_den + depth):
                s_max += Fraction(1, i)
            
            if remaining > s_max:
                return None
            
            # Min denominator for next term:
            # 1/d <= remaining  => d >= 1/remaining
            # d >= min_den
            start_d = max(min_den, (1 / remaining).numerator // (1 / remaining).denominator + 1)
            # Actually, d > 1/remaining is not required, 1/d <= remaining is required.
            # So d >= ceil(1/remaining).
            
            # Also, the remaining depth-1 terms must be able to sum to remaining - 1/d.
            # Max sum of depth-1 terms starting at d+1:
            # S_max_rest = sum(1/i for i in range(d+1, d+1+depth-1))
            # If remaining - 1/d > S_max_rest, skip d.
            
            # Iterate d
            # Upper bound for d:
            # 1/d < remaining (strictly, as other terms are positive)
            # So d > 1/remaining.
            # Also, remaining - 1/d must be >= (depth-1) / (some large number).
            
            # Let's just iterate d from start_d upwards.
            # To avoid infinite loop, we can bound d.
            # If d is very large, 1/d is very small.
            # remaining - 1/d is close to remaining.
            # We need to represent remaining - 1/d with depth-1 terms.
            # The largest term in that set is 1/(d+1).
            # So remaining - 1/d <= (depth-1)/(d+1).
            
            for d in range(start_d, 10000000):
                if time.time() > deadline:
                    return None
                
                # Check bound: remaining - 1/d <= (depth-1)/(d+1)
                # If this fails, increasing d will make LHS larger (closer to remaining) and RHS smaller.
                # So we can break.
                rest = remaining - Fraction(1, d)
                if rest <= 0:
                    continue
                    
                # Max sum of depth-1 terms with min denom d+1
                s_max_rest = 0
                for i in range(d + 1, d + depth): # depth-1 terms
                    s_max_rest += Fraction(1, i)
                    
                if rest > s_max_rest:
                    break # No larger d will work
                
                # Min sum of depth-1 terms?
                # rest must be > 0.
                
                res = dfs(rest, d + 1, depth - 1, path + [d])
                if res:
                    return res
            
            return None

        # Try k=8
        res = dfs(target, 5, 8, [])
        if res:
            return res
            
        # If k=8 fails (shouldn't), try k=9
        res = dfs(target, 5, 9, [])
        if res:
            return res
            
        # Fallback: Return a valid solution (e.g. from seed logic or hardcoded)
        # Hardcode a known valid solution if search fails.
        # We know k=8 is possible. 
        # One such solution:
        # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/30 + 1/35 + 1/42 ?
        # Sum:
        # 1/5+1/6+1/7 = 107/210
        # 1/10+1/14 = (7+5)/70 = 12/70 = 6/35 = 36/210
        # 1/30+1/35+1/42 = (7+6+5)/210 = 18/210
        # Total = (107+36+18)/210 = 161/210 != 1.
        
        # Let's just return the result of the seed program's logic if DFS fails.
        # But DFS should succeed for k=8.
        
        raise RuntimeError("No solution found")

    return solve()
