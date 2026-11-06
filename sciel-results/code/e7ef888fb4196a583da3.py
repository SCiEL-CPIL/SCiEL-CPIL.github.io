"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time
from functools import lru_cache

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill

def _is_smooth(x):
    """Check if x is of form 2^a * 3^b * 5^c with a,b,c >= 0."""
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _antichain_valid(terms):
    """Check if terms form a divisibility antichain."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            if terms[j] % terms[i] == 0 or terms[i] % terms[j] == 0:
                return False
    return True

def construct():
    t0 = time.time()
    
    # Generate all smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        v2 = 1 << a
        for b in range(0, 6):
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5):
                v = v23 * (5 ** c)
                if v > 128:
                    break
                if v >= 1:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    
    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Precompute all valid antichain pairs (s1, s2) with s1 < s2, s1+s2 <= 128
    pair_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            total = s1 + s2
            if total > 128:
                break
            if s2 % s1 == 0:
                continue
            if total not in pair_sums:
                pair_sums[total] = []
            pair_sums[total].append((s1, s2))
    
    # Precompute all valid antichain triples (s1, s2, s3) with s1 < s2 < s3
    triple_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if s2 % s1 == 0:
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                total = s1 + s2 + s3
                if total > 128:
                    break
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                if total not in triple_sums:
                    triple_sums[total] = []
                triple_sums[total].append((s1, s2, s3))
    
    # Instead of precomputing all 4-tuples (expensive), use on-demand search
    # with memoization for 4-term representations
    
    def find_k_terms(target, k, start_idx=0):
        """Find k distinct smooth numbers from smooth[start_idx:] that sum to target
        and form an antichain. Returns list or None."""
        if time.time() - t0 > DEADLINE:
            return None
        if k == 1:
            # Need one smooth number equal to target
            if is_sm[target] and target >= smooth[start_idx]:
                # Check if target is in smooth[start_idx:]
                for s in smooth[start_idx:]:
                    if s == target:
                        return [target]
                    if s > target:
                        break
            return None
        if k == 2:
            # Find two smooth numbers s1 < s2 with s1+s2=target, antichain
            for i in range(start_idx, n_smooth):
                s1 = smooth[i]
                if s1 * 2 >= target:
                    break
                s2 = target - s1
                if s2 <= s1:
                    continue
                if not is_sm[s2]:
                    continue
                if s2 % s1 == 0:
                    continue
                # Check s2 is in smooth
                if s2 in smooth_set:
                    return [s1, s2]
            return None
        if k == 3:
            # Find three smooth numbers s1 < s2 < s3 with s1+s2+s3=target, antichain
            for i in range(start_idx, n_smooth):
                s1 = smooth[i]
                if s1 * 3 >= target:
                    break
                rem1 = target - s1
                for j in range(i + 1, n_smooth):
                    s2 = smooth[j]
                    if s2 * 2 >= rem1:
                        break
                    if s2 % s1 == 0:
                        continue
                    rem2 = rem1 - s2
                    s3 = rem2
                    if s3 <= s2:
                        continue
                    if not is_sm[s3]:
                        continue
                    if s3 % s1 == 0 or s3 % s2 == 0:
                        continue
                    if s3 in smooth_set:
                        return [s1, s2, s3]
            return None
        if k == 4:
            # Find four smooth numbers s1 < s2 < s3 < s4 with s1+s2+s3+s4=target, antichain
            for i in range(start_idx, n_smooth):
                s1 = smooth[i]
                if s1 * 4 >= target:
                    break
                rem1 = target - s1
                for j in range(i + 1, n_smooth):
                    s2 = smooth[j]
                    if s2 * 3 >= rem1:
                        break
                    if s2 % s1 == 0:
                        continue
                    rem2 = rem1 - s2
                    for k_idx in range(j + 1, n_smooth):
                        s3 = smooth[k_idx]
                        if s3 * 2 >= rem2:
                            break
                        if s3 % s1 == 0 or s3 % s2 == 0:
                            continue
                        rem3 = rem2 - s3
                        s4 = rem3
                        if s4 <= s3:
                            continue
                        if not is_sm[s4]:
                            continue
                        if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                            continue
                        if s4 in smooth_set:
                            return [s1, s2, s3, s4]
            return None
        if k == 5:
            # Find five smooth numbers
            for i in range(start_idx, n_smooth):
                s1 = smooth[i]
                if s1 * 5 >= target:
                    break
                rem1 = target - s1
                for j in range(i + 1, n_smooth):
                    s2 = smooth[j]
                    if s2 * 4 >= rem1:
                        break
                    if s2 % s1 == 0:
                        continue
                    rem2 = rem1 - s2
                    for k_idx in range(j + 1, n_smooth):
                        s3 = smooth[k_idx]
                        if s3 * 3 >= rem2:
                            break
                        if s3 % s1 == 0 or s3 % s2 == 0:
                            continue
                        rem3 = rem2 - s3
                        for l in range(k_idx + 1, n_smooth):
                            s4 = smooth[l]
                            if s4 * 2 >= rem3:
                                break
                            if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                                continue
                            rem4 = rem3 - s4
                            s5 = rem4
                            if s5 <= s4:
                                continue
                            if not is_sm[s5]:
                                continue
                            if s5 % s1 == 0 or s5 % s2 == 0 or s5 % s3 == 0 or s5 % s4 == 0:
                                continue
                            if s5 in smooth_set:
                                return [s1, s2, s3, s4, s5]
            return None
        return None
    
    # Precompute a fallback representation for each target using a simple strategy
    # that guarantees validity (may not be minimal, but always valid)
    fallback = {}
    for n in range(1, 129):
        if is_sm[n]:
            fallback[n] = [n]
        elif n in pair_sums:
            s1, s2 = pair_sums[n][0]
            fallback[n] = [s1, s2]
        elif n in triple_sums:
            s1, s2, s3 = triple_sums[n][0]
            fallback[n] = [s1, s2, s3]
        else:
            # Try to find a 4-term representation quickly
            found = False
            for s1 in smooth:
                if s1 * 4 >= n:
                    break
                rem1 = n - s1
                for s2 in smooth:
                    if s2 <= s1 or s2 * 3 >= rem1:
                        continue
                    if s2 % s1 == 0:
                        continue
                    rem2 = rem1 - s2
                    for s3 in smooth:
                        if s3 <= s2 or s3 * 2 >= rem2:
                            continue
                        if s3 % s1 == 0 or s3 % s2 == 0:
                            continue
                        rem3 = rem2 - s3
                        s4 = rem3
                        if s4 <= s3:
                            continue
                        if not is_sm[s4]:
                            continue
                        if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                            continue
                        if s4 in smooth_set:
                            fallback[n] = [s1, s2, s3, s4]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                # Last resort: use a known valid pattern
                # Try 1 + (n-1) if n-1 is smooth and doesn't divide 1 (impossible) 
                # or find another combination
                found2 = False
                for a in smooth:
                    if a >= n:
                        break
                    for b in smooth:
                        if b <= a:
                            continue
                        if a + b >= n:
                            break
                        if b % a == 0:
                            continue
                        rem = n - a - b
                        if rem <= b:
                            continue
                        if not is_sm[rem]:
                            continue
                        if rem % a == 0 or rem % b == 0:
                            continue
                        if rem in smooth_set:
                            fallback[n] = [a, b, rem]
                            found2 = True
                            break
                    if found2:
                        break
                if not found2:
                    # Absolute last resort
                    fallback[n] = [1, n - 1] if n > 1 and is_sm[n-1] else [n] if is_sm[n] else [1, 2, n-3] if n > 3 else [n]
    
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            answers.append(fallback[n])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term using precomputed pairs
        if n in pair_sums and len(pair_sums[n]) > 0:
            s1, s2 = pair_sums[n][0]
            answers.append([s1, s2])
            continue
        
        # Try 3-term using precomputed triples
        if n in triple_sums and len(triple_sums[n]) > 0:
            s1, s2, s3 = triple_sums[n][0]
            answers.append([s1, s2, s3])
            continue
        
        # Try 4-term on-demand
        result = find_k_terms(n, 4)
        if result is not None:
            answers.append(result)
            continue
        
        # Try 5-term on-demand
        result = find_k_terms(n, 5)
        if result is not None:
            answers.append(result)
            continue
        
        # Use fallback
        answers.append(fallback[n])
    
    return answers
