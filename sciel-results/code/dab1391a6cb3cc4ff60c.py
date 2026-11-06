"""Optimized solver for large reciprocal weight under LCM separation."""
# EVOLVE-BLOCK-START
import math
from math import gcd

def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    """
    MAX_N = 256
    
    # Precompute divisors for each number in [2, MAX_N]
    # divisors[i] is a set of divisors of i (including i itself)
    divisors = [set() for _ in range(MAX_N + 1)]
    for d in range(2, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            divisors[m].add(d)
    
    # Greedy algorithm:
    # We want to maximize sum(1/x).
    # The constraint is that for any two selected numbers a, b, lcm(a, b) > 256.
    # lcm(a, b) = a * b / gcd(a, b).
    # If lcm(a, b) <= 256, then a and b cannot both be in the set.
    
    # Key insight:
    # If we select a number `x`, we cannot select any other number `y` such that lcm(x, y) <= 256.
    # Note that lcm(x, y) <= 256 implies that x and y share enough structure or are small enough.
    
    # Let's sort candidates by reciprocal (descending) i.e., by value (ascending).
    # We will try to add numbers in order of smallest value first (largest reciprocal).
    
    selected = []
    
    # To efficiently check compatibility, we can maintain a set of "forbidden" numbers.
    # When we add `x` to `selected`, any `y` in [2, 256] such that lcm(x, y) <= 256 becomes forbidden.
    # Actually, we just need to check if a new candidate `y` is compatible with all `x` in `selected`.
    
    # Let's precompute the set of numbers incompatible with each `x`.
    # incompatible[x] = set of y in [2, 256] such that lcm(x, y) <= 256.
    
    incompatible = [set() for _ in range(MAX_N + 1)]
    for x in range(2, MAX_N + 1):
        for y in range(2, MAX_N + 1):
            if x == y:
                continue
            # Check lcm(x, y) <= 256
            g = gcd(x, y)
            if x // g * y <= 256: # This is (x * y) / g <= 256
                incompatible[x].add(y)
    
    # Now, greedy selection
    # Start with empty set.
    # Try adding candidates from 2 to 256.
    # A candidate `c` is valid if for all `s` in `selected`, `c` is not in `incompatible[s]` (which is symmetric).
    # Actually, `c` is incompatible with `s` if `s` is in `incompatible[c]` or `c` is in `incompatible[s]`.
    # Since lcm is symmetric, incompatible[s] contains c iff incompatible[c] contains s.
    
    # We can just keep `selected` and check each new candidate against all existing ones.
    # With MAX_N=256, the size of `selected` will be small, so O(N^2) is fine.
    
    for c in range(2, MAX_N + 1):
        # Check if c can be added
        can_add = True
        for s in selected:
            if c in incompatible[s]:
                can_add = False
                break
        if can_add:
            selected.append(c)
            
    # The greedy algorithm from smallest to largest might not be optimal.
    # Let's analyze the structure.
    # Numbers > 128:
    # If x > 128 and y > 128, then x*y > 128*128 = 16384.
    # lcm(x, y) >= max(x, y) > 128.
    # Is lcm(x, y) always > 256?
    # lcm(x, y) = x*y/gcd(x,y).
    # If x and y are both > 128, let x = 128 + a, y = 128 + b.
    # If they are coprime, lcm is huge.
    # If they share a factor, say 2.
    # Let x=129 (odd), y=131 (odd). gcd=1. lcm > 16000.
    # Let x=130, y=132. gcd(130, 132) = 2. lcm = 130*132/2 = 8580 > 256.
    # What is the minimum lcm for two numbers > 128?
    # We want to minimize x*y/gcd(x,y) subject to x,y > 128.
    # To minimize the product, we pick the smallest numbers: 129, 130, ...
    # If we pick 129 and 130: gcd=1, lcm=16770.
    # If we pick 128 and 129: 128 is not > 128.
    # Consider x=128, y=129. 128 is not in the set if we only consider >128.
    # But what if we include 128?
    # lcm(128, 129) = 128*129 = 16512 > 256.
    # lcm(128, 130) = 128*130/2 = 8320 > 256.
    # It seems that ANY pair of distinct numbers from [128, 256] has lcm > 256.
    # Let's verify.
    # Let x, y in [128, 256], x != y.
    # We need lcm(x, y) > 256.
    # lcm(x, y) >= max(x, y) >= 129 (if both >= 129).
    # If one is 128, say x=128. y in [129, 256].
    # lcm(128, y) = 128 * y / gcd(128, y).
    # 128 = 2^7.
    # If y is odd, gcd=1, lcm = 128*y >= 128*129 > 256.
    # If y is even, let y = 2^k * m with m odd, k>=1.
    # gcd(128, y) = 2^min(7, k).
    # If k >= 7, gcd = 128, lcm = y. Since y >= 128, lcm >= 128.
    # Wait. If y is a multiple of 128, say y=256.
    # lcm(128, 256) = 256.
    # The condition is lcm > 256.
    # So lcm(128, 256) = 256 is NOT > 256.
    # So 128 and 256 cannot be together.
    # What about 128 and 192? 192 = 64 * 3. gcd(128, 192) = 64. lcm = 128*192/64 = 384 > 256.
    # So 128 is compatible with most numbers > 128, except multiples of 128 (only 256).
    # So we can include 128, but not 256? Or include 256 but not 128?
    # 1/128 + 1/256 vs just 1/128?
    # If we include 128, we lose 256. Gain 1/128, lose 1/256. Net gain 1/256.
    # So it's better to include 128 and exclude 256?
    # Or include both? No, they conflict.
    # Which one is "more expensive" to lose?
    # If we include 128, we must exclude 256.
    # If we include 256, we must exclude 128.
    # Sum with 128: 1/128. Sum with 256: 1/256.
    # 1/128 > 1/256. So prefer 128.
    
    # So, the set of all integers in [128, 256] is almost a valid clique, except for the pair (128, 256).
    # Let's check if there are other conflicts in [128, 256].
    # We need lcm(x, y) <= 256.
    # If x, y >= 128, lcm(x, y) <= 256 implies lcm(x, y) = 128 or 256?
    # Since x, y divide lcm(x, y), x and y must be divisors of 128 or 256.
    # Divisors of 256: 1, 2, 4, ..., 128, 256.
    # In range [128, 256], the only divisors of 256 are 128 and 256.
    # So the only pair in [128, 256] that has lcm <= 256 is (128, 256).
    # Thus, the set S1 = [128, 256] \ {256} is a valid clique.
    # Sum = sum(1/k for k in 128..255).
    
    # Can we add numbers < 128?
    # Let x < 128.
    # We need lcm(x, y) > 256 for all y in S1.
    # If we pick x=2.
    # We need lcm(2, y) > 256 for all y in S1.
    # lcm(2, y) = 2*y if y is odd, y if y is even.
    # If y is even and y <= 256, lcm(2, y) = y <= 256.
    # So 2 conflicts with all even numbers in S1.
    # S1 contains many even numbers.
    # To include 2, we must exclude all even numbers in S1.
    # This loses a huge amount of reciprocal sum.
    # Gain 1/2. Loss sum(1/y for even y in 128..255).
    # This loss is roughly half of the sum of S1.
    # Sum(S1) is approx ln(255/128) ~ 0.68.
    # Half of that is ~ 0.34.
    # 1/2 = 0.5.
    # Net gain ~ 0.16.
    # It might be worth it?
    # Let's check the greedy result again.
    # The greedy algorithm added 2 first.
    # Then it would have skipped all even numbers.
    # Then it would have added odds?
    # Let's trace the greedy algorithm provided in the current program.
    # It iterates x from 2 to 256.
    # x=2: selected=[2].
    # x=3: lcm(2,3)=6<=256. Skip.
    # x=4: lcm(2,4)=4<=256. Skip.
    # ...
    # x=129: lcm(2,129) = 258 > 256. Add. selected=[2, 129].
    # x=130: lcm(2,130) = 130 <= 256. Skip.
    # x=131: lcm(2,131) = 262 > 256. Add. selected=[2, 129, 131].
    # ...
    # It will add all ODD numbers in [129, 255].
    # And 2.
    # What about 128?
    # x=128: lcm(2,128) = 128 <= 256. Skip.
    # So the greedy solution is {2} U {odd in [129, 255]}.
    
    # Let's calculate the score of this set vs S1 = [128, 255].
    # Score(Greedy) = 1/2 + sum(1/k for k in 129..255, k odd).
    # Score(S1) = sum(1/k for k in 128..255).
    
    # Let's estimate.
    # Sum(1/k for k in 128..255) ~ ln(255/128) + (1/255 - 1/128) approx?
    # More accurately, H_255 - H_127.
    # H_255 ~ 6.13. H_127 ~ 5.43. Diff ~ 0.70.
    # Sum(odds in 129..255) ~ half of that? ~ 0.35.
    # Score(Greedy) ~ 0.5 + 0.35 = 0.85.
    # Score(S1) ~ 0.70.
    # So Greedy is better!
    
    # Can we do even better?
    # What if we pick {2, 3}?
    # lcm(2,3)=6 <= 256. Conflict.
    # What if we pick {2, 5}?
    # lcm(2,5)=10 <= 256. Conflict.
    # Any odd prime p. lcm(2, p) = 2p.
    # If 2p <= 256, i.e., p <= 128, then 2 and p conflict.
    # So 2 conflicts with all odds <= 128.
    # 2 is compatible with odds > 128.
    # This matches the greedy result.
    
    # Is there a better combination?
    # What if we don't pick 2?
    # Then we can pick 3, 4, etc?
    # If we don't pick 2, we can pick all evens in S1.
    # S1 already has all evens and odds in 128..255.
    # If we don't pick 2, we can just take S1.
    # Score(S1) ~ 0.70.
    # Score({2} U Odds) ~ 0.85.
    # So picking 2 is better.
    
    # What if we pick 4 instead of 2?
    # 4 conflicts with all multiples of 4.
    # lcm(4, y) <= 256.
    # If y is multiple of 4, lcm=y.
    # If y is 2*odd, lcm=2y.
    # This seems worse than 2 because 1/4 < 1/2 and it conflicts with more things (or same).
    
    # What if we pick 2 and some other small number that is compatible?
    # We established 2 is only compatible with odds > 128 (and 1, but 1 is excluded).
    # So we can't add any other small number to {2}.
    
    # What if we pick a set of small numbers that are pairwise compatible?
    # e.g. {3, 5, 7, ...}
    # lcm(3,5)=15.
    # lcm(3,7)=21.
    # lcm(5,7)=35.
    # They are pairwise compatible if lcm > 256?
    # No, lcm(3,5)=15 <= 256. Conflict.
    # So we can't pick 3 and 5 together.
    # In fact, for any x, y < 16, lcm(x,y) is usually small.
    # To have lcm(x,y) > 256 with x,y small, they must be coprime and product > 256.
    # e.g. 17 and 19. lcm=323 > 256.
    # So we could pick {17, 19, 23, ...}
    # But 1/17 + 1/19 ... is much smaller than 1/2.
    
    # So the main competition is between:
    # 1. {2} U {Odds in 129..255}
    # 2. [128, 255] (excluding 256)
    # 3. Maybe {2, 3, ...} if we can find a clique of small numbers?
    #    But small numbers have small lcms.
    #    Max clique size of small numbers with lcm > 256 is likely 1 or 2.
    #    If we pick {2}, we gain 0.5 but lose evens in 128..255.
    #    If we pick nothing small, we gain evens in 128..255.
    
    # Let's refine the "Greedy" result.
    # The greedy algorithm I ran mentally picked 2, then odds in 129..255.
    # Did it pick 128? No, because 128 is even and lcm(2,128)=128<=256.
    # Did it pick any even number > 128?
    # Let y be even, y > 128.
    # lcm(2, y) = y.
    # If y <= 256, lcm <= 256.
    # So 2 conflicts with ALL even numbers in [2, 256].
    # So if 2 is selected, NO other even number can be selected.
    # So the set must be {2} U {Odds compatible with 2}.
    # Odds compatible with 2: lcm(2, odd) = 2*odd.
    # 2*odd > 256 => odd > 128.
    # So odds in {129, 131, ..., 255}.
    # Are these odds pairwise compatible?
    # Let a, b be distinct odds in [129, 255].
    # lcm(a, b) > 256?
    # a, b >= 129.
    # lcm(a, b) >= max(a, b) >= 129.
    # Is it possible lcm(a, b) <= 256?
    # If a divides b, lcm=b.
    # If a=129, b=129*2=258 (not in range).
    # Max odd in range is 255.
    # 129 * 2 = 258.
    # So no odd in range is a multiple of another odd in range (since 2*129 > 255).
    # So gcd(a,b) < a.
    # lcm(a,b) = a*b/gcd(a,b) > a*b/a = b >= 129.
    # Wait, we need > 256.
    # If gcd(a,b)=1, lcm=a*b >= 129*131 > 16000.
    # If gcd(a,b)>1, let g=gcd(a,b). a=g*a', b=g*b'.
    # lcm = g*a'*b'.
    # Since a,b are odd, g is odd.
    # If g >= 3, a' >= 1, b' >= 1.
    # a = g*a' >= 129 => a' >= 43 (if g=3).
    # b = g*b' >= 129 => b' >= 43.
    # lcm = g*a'*b' >= 3*43*43 = 5547 > 256.
    # So yes, all pairs of odds in [129, 255] are compatible.
    
    # So the candidate set A = {2} U {odd x | 129 <= x <= 255} is valid.
    # Score(A) = 1/2 + sum(1/x for x in {129, 131, ..., 255}).
    
    # Let's check if we can improve by swapping 2 with something else?
    # No, 2 is the best single element.
    # Can we add more elements to A?
    # No, all evens are blocked by 2.
    # All odds < 129 are blocked by 2 (lcm(2, odd) = 2*odd <= 256).
    # So A is a maximal set containing 2.
    
    # What if we don't use 2?
    # Then we can use all of [128, 255] (except 256 if 128 is there? No, 128 and 256 conflict).
    # Set B = [128, 255].
    # Score(B) = sum(1/x for x in 128..255).
    # We compared A and B. A wins.
    
    # Is there a set C that is better than A?
    # Maybe a set that doesn't include 2, but includes some other small number that allows keeping more of the high range?
    # E.g., pick 3.
    # 3 conflicts with multiples of 3.
    # lcm(3, y) <= 256.
    # If y is multiple of 3, lcm=y.
    # If y is not multiple of 3, lcm=3y/gcd(3,y).
    # If y not div by 3, gcd=1, lcm=3y.
    # 3y <= 256 => y <= 85.
    # So 3 conflicts with all y <= 85 (not div by 3) and all y <= 256 (div by 3).
    # Essentially 3 conflicts with almost everything small and medium.
    # It allows keeping numbers > 85 that are not multiples of 3?
    # Let's check y=100. lcm(3,100)=300>256. OK.
    # y=102. lcm(3,102)=102<=256. Conflict.
    # So 3 blocks all multiples of 3.
    # And blocks all y < 85.
    # So if we pick 3, we lose all multiples of 3 in [128, 255].
    # And we gain 1/3.
    # 1/3 = 0.33.
    # Loss: sum(1/x for x in [128, 255] if x%3==0).
    # Approx 1/3 of the sum of B.
    # Sum(B) ~ 0.70. Loss ~ 0.23.
    # Net gain ~ 0.10.
    # Compare to picking 2: Net gain ~ 0.15.
    # So 2 is better than 3.
    
    # It seems the greedy approach of picking the smallest possible number is optimal here.
    # The current program implements exactly this.
    # Why did I think I needed to rewrite it?
    # The task says "Rewrite the program from scratch with a stronger algorithm."
    # Maybe the greedy is not globally optimal?
    # What if we pick {2, 129, 131, ...}
    # Is there a set like {4, 129, ...} that is better?
    # 1/4 = 0.25. 1/2 = 0.5.
    # 4 blocks more (multiples of 4, and y where lcm(4,y)<=256).
    # lcm(4, y) <= 256.
    # If y div by 4, lcm=y.
    # If y=2*odd, lcm=2y.
    # If y odd, lcm=4y.
    # 4y <= 256 => y <= 64.
    # So 4 blocks all y <= 64, all multiples of 4, and 2*odd where 2y<=256 => y<=128.
    # So 4 blocks all evens <= 128 and all odds <= 64.
    # It allows keeping odds > 64?
    # Check y=100 (even, not div by 4). lcm(4,100) = 100. Conflict.
    # So 4 blocks all evens.
    # Same as 2?
    # 2 blocks all evens.
    # 4 blocks all evens.
    # But 1/4 < 1/2.
    # So 2 is strictly better than 4.
    
    # So the algorithm "Pick 2, then pick all compatible numbers in ascending order" is likely optimal.
    # The current program does this.
    # To make it "stronger" or "distinct", I can implement a local search to verify or improve.
    # Or just implement the same logic but with explicit validation and perhaps a better structure.
    # Since I must not reuse the current design, I will implement a local search starting from the greedy solution.
    
    # 1. Generate Greedy Solution G.
    # 2. Try to add any number from [2, 256] not in G.
    #    - If compatible, add it.
    # 3. Try to swap each element in G with any number not in G.
    #    - If swap improves score and is valid, do it.
    # 4. Repeat until no improvement.
    
    # This is a robust approach.
    
    selected = []
    for c in range(2, MAX_N + 1):
        if all(c not in incompatible[s] for s in selected):
            selected.append(c)
            
    # Local Search
    improved = True
    while improved:
        improved = False
        # Try adding
        for c in range(2, MAX_N + 1):
            if c not in selected:
                if all(c not in incompatible[s] for s in selected):
                    selected.append(c)
                    improved = True
                    break # Restart loop to update incompatibility checks?
                         # Actually 'selected' changes, so we break and restart while loop.
                         # But we just added, so we can continue checking?
                         # No, adding one might block others, but we want to maximize, so adding is always good if valid.
                         # We should keep adding until we can't.
        # Try swapping
        if not improved:
            for i, s in enumerate(selected):
                best_swap = None
                best_gain = 0
                for c in range(2, MAX_N + 1):
                    if c in selected:
                        continue
                    # Check if c can replace s
                    # New set: selected without s, plus c
                    # Must be valid: c compatible with all others in selected (excluding s)
                    valid = True
                    for j, s2 in enumerate(selected):
                        if j == i:
                            continue
                        if c in incompatible[s2]:
                            valid = False
                            break
                    if valid:
                        # Calculate gain
                        # Gain = 1/c - 1/s
                        if 1/c > 1/s:
                            gain = 1/c - 1/s
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = c
                if best_swap is not None:
                    selected[i] = best_swap
                    improved = True
                    break # Restart to recalculate
            
    return selected

# EVOLVE-BLOCK-END
