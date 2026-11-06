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
    
    # Precompute all valid antichain pairs
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
    
    # Precompute all valid antichain triples
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
    
    # Precompute all valid antichain 4-tuples
    quad_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if s2 % s1 == 0:
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    total = s1 + s2 + s3 + s4
                    if total > 128:
                        break
                    if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                        continue
                    if total not in quad_sums:
                        quad_sums[total] = []
                    quad_sums[total].append((s1, s2, s3, s4))
    
    # Precompute all valid antichain 5-tuples
    quint_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if s2 % s1 == 0:
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                        continue
                    for m in range(l + 1, n_smooth):
                        s5 = smooth[m]
                        total = s1 + s2 + s3 + s4 + s5
                        if total > 128:
                            break
                        if s5 % s1 == 0 or s5 % s2 == 0 or s5 % s3 == 0 or s5 % s4 == 0:
                            continue
                        if total not in quint_sums:
                            quint_sums[total] = []
                        quint_sums[total].append((s1, s2, s3, s4, s5))
    
    # Build answer for each target
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: use a known valid representation
            if is_sm[n]:
                answers.append([n])
            elif n in pair_sums:
                s1, s2 = pair_sums[n][0]
                answers.append([s1, s2])
            elif n in triple_sums:
                s1, s2, s3 = triple_sums[n][0]
                answers.append([s1, s2, s3])
            elif n in quad_sums:
                s1, s2, s3, s4 = quad_sums[n][0]
                answers.append([s1, s2, s3, s4])
            elif n in quint_sums:
                s1, s2, s3, s4, s5 = quint_sums[n][0]
                answers.append([s1, s2, s3, s4, s5])
            else:
                # Absolute last resort - should not happen
                answers.append([n] if is_sm[n] else [1, n-1])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term
        if n in pair_sums and len(pair_sums[n]) > 0:
            s1, s2 = pair_sums[n][0]
            answers.append([s1, s2])
            continue
        
        # Try 3-term
        if n in triple_sums and len(triple_sums[n]) > 0:
            s1, s2, s3 = triple_sums[n][0]
            answers.append([s1, s2, s3])
            continue
        
        # Try 4-term
        if n in quad_sums and len(quad_sums[n]) > 0:
            s1, s2, s3, s4 = quad_sums[n][0]
            answers.append([s1, s2, s3, s4])
            continue
        
        # Try 5-term
        if n in quint_sums and len(quint_sums[n]) > 0:
            s1, s2, s3, s4, s5 = quint_sums[n][0]
            answers.append([s1, s2, s3, s4, s5])
            continue
        
        # Should not reach here for n<=128, but fallback
        if is_sm[n]:
            answers.append([n])
        elif n in pair_sums:
            s1, s2 = pair_sums[n][0]
            answers.append([s1, s2])
        elif n in triple_sums:
            s1, s2, s3 = triple_sums[n][0]
            answers.append([s1, s2, s3])
        else:
            answers.append([n])
    
    return answers
