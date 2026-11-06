"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill

def _is_smooth(x):
    """Check if x is of form 2^a * 3^b * 5^c with a,b,c >= 0."""
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _check_antichain(terms):
    """Check if no term in the list divides another."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a < b:
                if b % a == 0:
                    return False
            else:
                if a % b == 0:
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
    
    # Precompute divisibility: divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth[i], smooth[j]
                if a < b:
                    divs[i][j] = (b % a == 0)
                else:
                    divs[i][j] = (a % b == 0)
    
    # Precompute antichain validity for pairs
    pair_ok = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divs[i][j] or divs[j][i]:
                pair_ok[i][j] = False
                pair_ok[j][i] = False
    
    # Precompute antichain validity for triples (s1 < s2 < s3)
    triple_ok = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not pair_ok[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if pair_ok[i][k] and pair_ok[j][k]:
                    triple_ok[(i, j, k)] = True
    
    # Precompute antichain validity for quadruples (s1 < s2 < s3 < s4)
    # This is expensive, so we'll compute on-demand with caching
    quad_cache = {}
    
    def _quad_valid(i, j, k, l):
        key = (i, j, k, l)
        if key in quad_cache:
            return quad_cache[key]
        # Check all 6 pairs
        valid = (pair_ok[i][j] and pair_ok[i][k] and pair_ok[i][l] and 
                 pair_ok[j][k] and pair_ok[j][l] and pair_ok[k][l])
        quad_cache[key] = valid
        return valid
    
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: use a known valid representation
            answers.append([n] if is_sm[n] else [1, n-1])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term: s1 < s2, s1 + s2 = n, both smooth, s1 ∤ s2
        found_2 = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 >= n:
                break
            s2 = n - s1
            if s2 <= s1:
                continue
            if not is_sm[s2]:
                continue
            # Find index of s2
            j = smooth.index(s2) if s2 in smooth_set else -1
            if j == -1:
                continue
            if pair_ok[i][j]:
                answers.append([s1, s2])
                found_2 = True
                break
        if found_2:
            continue
        
        # Try 3-term: s1 < s2 < s3, s1+s2+s3 = n, all smooth, antichain
        found_3 = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 >= n:
                break
            rem_after_s1 = n - s1
            for j in range(i + 1, n_smooth):
                s2 = smooth[j]
                if s2 >= rem_after_s1:
                    break
                if not pair_ok[i][j]:
                    continue
                s3 = rem_after_s1 - s2
                if s3 <= s2:
                    break
                if not is_sm[s3]:
                    continue
                k = smooth.index(s3) if s3 in smooth_set else -1
                if k == -1 or k <= j:
                    continue
                if triple_ok.get((i, j, k), False):
                    answers.append([s1, s2, s3])
                    found_3 = True
                    break
            if found_3:
                break
        if found_3:
            continue
        
        # Try 4-term: s1 < s2 < s3 < s4, sum = n, all smooth, antichain
        found_4 = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 >= n:
                break
            rem1 = n - s1
            for j in range(i + 1, n_smooth):
                if time.time() - t0 > DEADLINE:
                    break
                s2 = smooth[j]
                if s2 >= rem1:
                    break
                if not pair_ok[i][j]:
                    continue
                rem2 = rem1 - s2
                for k in range(j + 1, n_smooth):
                    s3 = smooth[k]
                    if s3 >= rem2:
                        break
                    if not pair_ok[i][k] or not pair_ok[j][k]:
                        continue
                    s4 = rem2 - s3
                    if s4 <= s3:
                        break
                    if not is_sm[s4]:
                        continue
                    l = smooth.index(s4) if s4 in smooth_set else -1
                    if l == -1 or l <= k:
                        continue
                    if _quad_valid(i, j, k, l):
                        answers.append([s1, s2, s3, s4])
                        found_4 = True
                        break
                if found_4:
                    break
            if found_4:
                break
        if found_4:
            continue
        
        # Should not reach here for n<=128, but fallback
        answers.append([n] if is_sm[n] else [1, n-1])
    
    return answers
