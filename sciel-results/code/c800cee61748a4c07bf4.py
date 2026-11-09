"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
import sys

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _try_solution(denoms):
    """
    Validate a list of denominators and return the largest one if valid, else None.
    """
    if not denoms:
        return None
    if len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    
    return max(denoms)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known high-quality solution: [8, 31, 248] -> max 248
    known_sol = [8, 31, 248]
    best_max = _try_solution(known_sol)
    best_list = known_sol
    
    if best_max is None:
        return known_sol

    # Fast bounded search for better solutions
    def _search_bounded(rem, min_d, depth, limit_max, used):
        """
        Find Egyptian fraction of 'rem' using denominators >= min_d, all < limit_max,
        not in 'used', with at most MAX_TERMS - depth more terms.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if depth >= MAX_TERMS:
            return None
        
        # Lower bound: need 1/d <= rem for positive remainder (for first term)
        # Actually we can have 1/d > rem if we use multiple terms, but for efficiency
        # let's use the standard bound: d >= ceil(1/rem)
        min_d_eff = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if min_d_eff >= limit_max:
            return None
        
        # Upper bound on d: we need enough remaining terms to sum to rem - 1/d
        # With k terms left (after picking d), max sum is 1/(d+1) + 1/(d+2) + ... 
        # But simpler: if rem - 1/d > 0, we need at least 1 more term.
        # A practical upper bound: d cannot be too large or rem - 1/d becomes too small
        # to be represented with remaining terms under limit_max.
        
        # Also: if we pick d, then rem_new = rem - 1/d.
        # If rem_new > 0, we need to represent it with (MAX_TERMS - depth - 1) terms,
        # each >= d+1 and < limit_max.
        # The maximum possible sum with k terms each < limit_max is bounded by
        # k/limit_max (very loose). Better: the terms must be >= d+1, so max sum
        # is roughly k/(d+1) for small k, but this is also loose.
        
        # Key pruning: if rem_new > 0 and (MAX_TERMS - depth - 1) == 0, fail.
        # Also: the smallest denominator in remaining is d+1, so max sum with
        # k terms is at most k/(d+1) (very loose upper bound on what's achievable).
        # Actually that's an upper bound on the sum, not lower. We need a LOWER bound
        # on what's achievable to prune: the MINIMUM sum with k terms each >= d+1
        # and < limit_max is... well, it can be arbitrarily small. So no easy lower bound.
        
        # Better pruning: if rem_new > 0, we need at least one more term, so
        # MAX_TERMS - depth - 1 >= 1, i.e., depth <= MAX_TERMS - 2.
        # Already checked above.
        
        # Another bound: if rem_new > 0, the next denominator d2 must satisfy
        # d2 >= ceil(1/rem_new). If ceil(1/rem_new) >= limit_max, fail.
        # This is handled by the recursive call.
        
        # Iterate d from min_d_eff to limit_max - 1
        # Optimization: skip d if d is in used
        for d in range(min_d_eff, limit_max):
            if d in used:
                continue
            rem_new = rem - Fraction(1, d)
            if rem_new < 0:
                # Since d is increasing, 1/d is decreasing, rem_new is increasing.
                # So if rem_new < 0, it will become >= 0 for larger d.
                # But we want rem_new >= 0 (or we could allow negative and handle it,
                # but standard Egyptian fractions use positive remainders).
                # Actually if rem_new < 0, that means 1/d > rem, which is fine
                # only if we're not at the last term. But for simplicity, let's
                # continue searching (don't break).
                # Wait: if 1/d > rem, then rem - 1/d < 0. For the next term,
                # we'd need a negative reciprocal, which is not allowed.
                # So we MUST have 1/d <= rem, i.e., d >= ceil(1/rem).
                # But we already set min_d_eff = max(min_d, ceil(1/rem)).
                # So rem_new should be >= 0 for all d >= min_d_eff.
                # Let me double-check: ceil(1/rem) means d >= ceil(1/rem) => 1/d <= 1/ceil(1/rem) <= rem.
                # Yes, so rem_new >= 0 always. The check is redundant but safe.
                continue
            if rem_new == 0:
                return [d]
            
            # Additional pruning: if rem_new > 0, we need more terms.
            # The next min_d is d+1. Check if ceil(1/rem_new) < limit_max.
            next_min = d + 1
            next_min_eff = max(next_min, (rem_new.denominator + rem_new.numerator - 1) // rem_new.numerator)
            if next_min_eff >= limit_max:
                continue
            
            res = _search_bounded(rem_new, d + 1, depth + 1, limit_max, used | {d})
            if res is not None:
                return [d] + res
        return None

    # Search for solutions with max < best_max
    # Try different values of limit_max from best_max-1 down to find the minimum
    # More efficient: iterate limit_max from a reasonable lower bound up to best_max-1
    # and find the smallest limit_max for which a solution exists.
    
    # Theoretical lower bound: for 5/31, we need at least 3 terms (no 2-term solution exists).
    # With 3 terms 1/a + 1/b + 1/c = 5/31, a < b < c < limit_max.
    # The minimum possible max is at least... let's just search.
    
    # Strategy: binary search on limit_max, or just iterate from low to high.
    # Since best_max is 248, let's try to find if a solution exists with max < 248.
    # We'll try limit_max values from a lower bound up to best_max - 1.
    
    # Lower bound for max denominator: 
    # For 3 terms: 1/a + 1/b + 1/c = 5/31. If a=7, 1/7 = 0.1428..., 5/31=0.1612...
    # 5/31 - 1/7 = (35-31)/217 = 4/217. Need 1/b + 1/c = 4/217 with b>c>7.
    # 1/b + 1/c = (b+c)/(bc) = 4/217. 217(b+c) = 4bc. (4b-217)(4c-217) = 217^2 = 47089.
    # 47089 = 217^2 = (7*31)^2 = 49*961. Factors of 47089: 1, 7, 31, 49, 217, 961, 1517, 6781, 10127, 47089...
    # 4b-217 and 4c-217 must be positive factors (since b,c > 217/4 = 54.25).
    # So 4b-217 >= 1, meaning b >= 54.25, so b >= 55.
    # Let's check factor pairs of 47089 where both are > 0:
    # 47089 = 1 * 47089: 4b-217=1 -> b=54.5 (not int), 4b-217=47089 -> b=11826.5 (not int)
    # 47089 = 7 * 6727: 4b-217=7 -> b=56, 4c-217=6727 -> c=1736. So [7,56,1736] -> max=1736
    # 47089 = 31 * 1519: 4b-217=31 -> b=62, 4c-217=1519 -> c=434. So [7,62,434] -> max=434
    # 47089 = 49 * 961: 4b-217=49 -> b=66.5 (not int)
    # 47089 = 217 * 217: 4b-217=217 -> b=108.5 (not int)
    # So for a=7: best is [7,62,434] with max=434.
    
    # For a=8: 5/31 - 1/8 = (40-31)/248 = 9/248. Need 1/b+1/c=9/248, b>8.
    # 248(b+c)=9bc. (9b-248)(9c-248)=248^2=61504.
    # 61504 = 248^2 = (8*31)^2 = 64*961. Factors...
    # 9b-248 > 0 => b > 27.5, so b >= 28.
    # 61504 = 1*61504: 9b=249->b=27.78(no), 9b=61752->b=6861.33(no)
    # 61504 = 2*30752: 9b=250->b=27.78(no)
    # 61504 = 4*15376: 9b=252->b=28, 9c=15624->c=1736. [8,28,1736] max=1736
    # 61504 = 8*7688: 9b=256->b=28.44(no)
    # 61504 = 16*3844: 9b=264->b=29.33(no)
    # 61504 = 31*1984: 9b=279->b=31, 9c=2232->c=248. [8,31,248] max=248 ✓
    # 61504 = 64*961: 9b=312->b=34.67(no)
    # 61504 = 124*496: 9b=372->b=41.33(no)
    # 61504 = 248*248: 9b=496->b=55.11(no)
    # So for a=8: best is [8,31,248] max=248.
    
    # For a=9: 5/31-1/9 = (45-31)/279 = 14/279. 1/b+1/c=14/279, b>9.
    # 279(b+c)=14bc. (14b-279)(14c-279)=279^2=77841.
    # 14b>279 => b>19.9, so b>=20.
    # 77841 = 279^2 = (9*31)^2 = 81*961.
    # 77841 = 1*77841: 14b=280->b=20, 14c=78120->c=5580. [9,20,5580] max=5580
    # 77841 = 3*25947: 14b=282->b=20.14(no)
    # 77841 = 9*8649: 14b=288->b=20.57(no)
    # 77841 = 31*2511: 14b=310->b=22.14(no)
    # 77841 = 81*961: 14b=360->b=25.71(no)
    # 77841 = 243*320.33: not integer factor
    # 77841 = 279*279: 14b=558->b=39.86(no)
    # Hmm, let me factor 77841 properly. 279 = 9*31, so 77841 = 9^2 * 31^2 = 81*961.
    # Factors: 1,3,9,27,81,31,93,279,837,2511,7533,22599,961,2883,8649,25947,77841
    # Check pairs where 14b-279 and 14c-279 are both positive and 14b-279 <= 14c-279:
    # 1*77841: b=20, c=5580 -> max 5580
    # 3*25947: 14b=282->b=20.14 no
    # 9*8649: 14b=288->b=20.57 no
    # 27*2883: 14b=306->b=21.86 no
    # 31*2511: 14b=310->b=22.14 no
    # 81*961: 14b=360->b=25.71 no
    # 93*837: 14b=372->b=26.57 no
    # 279*279: 14b=558->b=39.86 no
    # 2883*27: already covered
    # So for a=9: only [9,20,5580], max=5580.
    
    # For a=10: 5/31-1/10=(50-31)/310=19/310. 1/b+1/c=19/310, b>10.
    # 310(b+c)=19bc. (19b-310)(19c-310)=310^2=96100.
    # 19b>310 => b>16.3, so b>=17.
    # 96100 = 310^2 = (10*31)^2 = 100*961.
    # Factors of 96100... this is getting complex. Let me just trust the search.
    
    # The search should find the optimum. Let me make it efficient.
    
    # Binary search approach: find the minimum M such that a solution exists with max <= M.
    # We know M=248 works. Let's try smaller M values.
    
    # Lower bound: for 3 terms, a >= 7 (since 1/6 > 5/31). 
    # If a=7, we showed max >= 434. If a=8, max >= 248.
    # Could there be a 4-term solution with smaller max?
    # E.g., [7, b, c, d] with d < 248?
    # 5/31 - 1/7 = 4/217. Need 1/b+1/c+1/d = 4/217 with 7<b<c<d<248.
    # This is a 3-term Egyptian fraction for 4/217.
    # 4/217 ≈ 0.01843. 1/55 ≈ 0.01818, 1/54 ≈ 0.01852.
    # So b is around 54-55.
    
    # Let me just run the search with better pruning.
    
    # Try decreasing limit_max from best_max-1 down to find the minimum.
    # More efficient: iterate limit_max from a lower bound up.
    
    # Let's try a direct search: for each possible max M from lower bound to best_max-1,
    # check if a solution exists. Use the bounded search.
    
    # Lower bound for M: at least 7 (but we need at least 3 terms, so M >= 9 at least)
    # Actually, for 3 terms a<b<c, we need c >= some value. Let's just start from 10.
    
    # To be efficient, let's try specific M values and use the search.
    # Start from a reasonable lower bound and go up.
    
    # Actually, let's just iterate M from best_max-1 down to 10 and find the first M
    # where a solution exists. But that's many iterations.
    
    # Better: iterate the first denominator a, and for each a, find the minimum
    # possible max denominator for the remaining terms.
    
    # For each a in [7, best_max-1]:
    #   rem = 5/31 - 1/a
    #   Find minimum M such that rem can be written as sum of reciprocals of
    #   distinct integers in [a+1, M-1] (at most MAX_TERMS-1 terms).
    #   The overall max is max(a, M-1) = M-1 (since M-1 > a for M > a+1).
    #   Update best if M-1 < best_max.
    
    # This is still expensive. Let me just run the bounded search with limit_max
    # decreasing from best_max-1.
    
    for limit_max in range(best_max - 1, 9, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try each possible first denominator a
        for a in range(7, limit_max):
            if time.time() > deadline:
                break
            rem_a = TARGET - Fraction(1, a)
            if rem_a <= 0:
                continue
            # Check if rem_a can be represented with denominators in [a+1, limit_max-1]
            res = _search_bounded(rem_a, a + 1, 1, limit_max, {a})
            if res is not None:
                candidate = [a] + res
                cand_max = _try_solution(candidate)
                if cand_max is not None and cand_max < best_max:
                    best_max = cand_max
                    best_list = candidate
                    found = True
                    break  # Found a solution with this limit_max, move to smaller limit_max
        
        if found:
            # We found a better solution, continue searching for even better
            # (with smaller limit_max)
            pass
        else:
            # No solution found with this limit_max, try smaller
            pass
    
    return best_list

# EVOLVE-BLOCK-END
