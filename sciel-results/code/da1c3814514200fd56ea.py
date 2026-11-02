"""Football pool covering code K_3(7,1). Direct-sum construction + optimized local search."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0


def _direct_sum_construct(q, n, radius):
    """Construct a covering code using direct sum of smaller codes.
    
    For q=3, n=7, radius=1: split into 4+3 coordinates.
    Use a perfect code (or near-perfect) for each part, then take the product.
    
    A Hamming code over GF(3) of length 4 has 3^2=9 codewords covering radius 1.
    Actually, let's use a simpler approach: 
    - For n=4, q=3, r=1: the Hamming code has (3^4-1)/(3-1) = 40/2 = 20... that's not right.
    - Let me think: a covering code of length 4 over GF(3) with radius 1 needs at least 
      3^4 / (1 + 4*2) = 81/9 = 9 codewords. The perfect code doesn't exist for these params.
    
    Better approach: use known small covering codes.
    For q=3, n=1, r=1: need 3 codewords (all of them).
    For q=3, n=2, r=1: need 3 codewords? No, each covers 1+2*2=5 words. 3^2=9, so ceil(9/5)=2? 
      But 2 codewords cover at most 10 words (with overlap). Let's check: codewords 00 and 11.
      00 covers: 00,01,02,10,20. 11 covers: 11,10,12,21,01. Union: 00,01,02,10,20,11,12,21. Missing 22.
      So need 3. E.g., 00,11,22 covers all 9.
    
    For q=3, n=3, r=1: 27 words, each covers 1+3*2=7. Lower bound ceil(27/7)=4.
      Try 4 codewords: 000,111,222, and one more? 000 covers 000,001,002,010,020,100,200.
      This is getting complex. Let me use a greedy approach for the small parts.
    """
    base = q
    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * base
    
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % base
            x //= base
        return d
    
    def to_int(d):
        x = 0
        for i in range(n):
            x += d[i] * power[i]
        return x
    
    # For the direct sum, we'll split n into parts and construct codes for each part.
    # Then the product code has size = product of sizes.
    # For n=7: split as 3+4 or 4+3.
    
    # Let's use a greedy construction for each part to get small covering codes.
    def greedy_cover(part_n, part_q, part_r):
        """Greedy covering code for part_n coordinates over part_q alphabet with radius part_r."""
        total_p = part_q ** part_n
        power_p = [1] * part_n
        for i in range(1, part_n):
            power_p[i] = power_p[i - 1] * part_q
        
        # Precompute balls
        balls_p = []
        for i in range(total_p):
            d = [0] * part_n
            tmp = i
            for pos in range(part_n):
                d[pos] = tmp % part_q
                tmp //= part_q
            b = [i]
            if part_r >= 1:
                for pos in range(part_n):
                    cur = d[pos]
                    for v in range(part_q):
                        if v != cur:
                            b.append(i + (v - cur) * power_p[pos])
            balls_p.append(set(b))
        
        uncovered = set(range(total_p))
        selected = []
        while uncovered:
            best = -1
            best_cnt = 0
            for c in range(total_p):
                cnt = len(uncovered & balls_p[c])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = c
            if best < 0:
                break
            selected.append(best)
            uncovered -= balls_p[best]
        return selected
    
    # Split n=7 into 3+4
    part1_n = 3
    part2_n = 4
    
    code1 = greedy_cover(part1_n, q, radius)
    code2 = greedy_cover(part2_n, q, radius)
    
    # Direct sum: each codeword is (c1, c2) where c1 in code1, c2 in code2
    # Convert to integer indices in the full space
    # c1 occupies positions 0..2, c2 occupies positions 3..6
    p1 = [1] * part1_n
    for i in range(1, part1_n):
        p1[i] = p1[i-1] * q
    p2 = [1] * part2_n
    for i in range(1, part2_n):
        p2[i] = p2[i-1] * q
    
    # Offset for part2 in the full space
    offset2 = q ** part1_n
    
    def c1_to_int(c1_idx):
        return c1_idx  # positions 0..2
    
    def c2_to_int(c2_idx):
        return c2_idx * q ** part1_n  # positions 3..6, but we need to shift digits
        # Actually c2_idx is in base q for part2_n digits, so:
        # full_idx = c1_idx + c2_idx * q^part1_n
        # But c2_idx's digits go into positions part1_n..n-1
        # c2_idx = d0 + d1*q + d2*q^2 + d3*q^3
        # We want: d0*q^3 + d1*q^4 + d2*q^5 + d3*q^6
        # = c2_idx * q^3 ... no, that's not right either.
        # c2_idx = d0 + d1*q + d2*q^2 + d3*q^3
        # c2_idx * q^3 = d0*q^3 + d1*q^4 + d2*q^5 + d3*q^6. Yes!
        return c2_idx * (q ** part1_n)
    
    full_code = []
    for c1 in code1:
        for c2 in code2:
            full_idx = c1_to_int(c1) + c2_to_int(c2)
            full_code.append(full_idx)
    
    return full_code


def construct(q, n, radius):
    start = time.time()
    total = q ** n  # 2187 for q=3,n=7
    base = q
    
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % base
            x //= base
        return d
    
    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * base
    
    # Precompute balls as numpy arrays
    balls_arr = []
    for i in range(total):
        d = digits_of(i)
        b = [i]
        for pos in range(n):
            cur = d[pos]
            for v in range(base):
                if v != cur:
                    b.append(i + (v - cur) * power[pos])
        balls_arr.append(np.array(b, dtype=np.int64))
    
    # Use direct-sum construction for initial solution
    if time.time() < start + _DEADLINE - _MARGIN:
        selected = _direct_sum_construct(q, n, radius)
    else:
        # Fallback to greedy
        uncovered = np.ones(total, dtype=bool)
        selected = []
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for c in range(total):
                cnt = np.count_nonzero(uncovered[balls_arr[c]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = c
            if best_idx < 0:
                break
            selected.append(best_idx)
            uncovered[balls_arr[best_idx]] = False
            if time.time() > start + _DEADLINE - _MARGIN:
                break
    
    # Verify coverage and complete if needed
    def build_cov(sel):
        cov = np.zeros(total, dtype=np.int64)
        for c in sel:
            cov[balls_arr[c]] += 1
        return cov
    
    cov = build_cov(selected)
    if cov.min() < 1:
        uncovered = cov < 1
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for c in range(total):
                cnt = np.count_nonzero(uncovered[balls_arr[c]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = c
            if best_idx < 0:
                break
            selected.append(best_idx)
            uncovered[balls_arr[best_idx]] = False
            if time.time() > start + _DEADLINE - _MARGIN:
                break
    
    best_list = list(selected)
    best_size = len(best_list)
    
    # Local search: removals and swaps
    improved = True
    while improved and (time.time() < start + _DEADLINE - _MARGIN):
        improved = False
        cov = build_cov(selected)
        
        # Pass 1: removals
        for k in range(len(selected) - 1, -1, -1):
            c = selected[k]
            if not (cov[balls_arr[c]] == 1).any():
                cov[balls_arr[c]] -= 1
                selected = selected[:k] + selected[k + 1:]
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_list = list(selected)
        
        # Pass 2: swaps
        if time.time() < start + _DEADLINE - _MARGIN:
            cov = build_cov(selected)
            for i in range(len(selected)):
                if time.time() > start + _DEADLINE - _MARGIN:
                    break
                ci = selected[i]
                if (cov[balls_arr[ci]] == 1).any():
                    # Find words uniquely covered by ci
                    uniq_words = np.where(cov == 1)[0]
                    ci_ball_set = set(balls_arr[ci].tolist())
                    uniq_in_ci = [w for w in uniq_words if w in ci_ball_set]
                    if not uniq_in_ci:
                        continue
                    # Try to find a replacement that covers all unique words
                    sel_set = set(selected)
                    found = False
                    for j in range(total):
                        if j in sel_set:
                            continue
                        j_ball_set = set(balls_arr[j].tolist())
                        if all(w in j_ball_set for w in uniq_in_ci):
                            # Perform swap
                            newlst = list(selected)
                            newlst[i] = j
                            ncov = build_cov(newlst)
                            # Try to remove redundant codewords
                            changed = True
                            while changed:
                                changed = False
                                for k in range(len(newlst) - 1, -1, -1):
                                    if not (ncov[balls_arr[newlst[k]]] == 1).any():
                                        ncov[balls_arr[newlst[k]]] -= 1
                                        newlst = newlst[:k] + newlst[k + 1:]
                                        changed = True
                            if len(newlst) < len(selected):
                                selected = newlst
                                improved = True
                                if len(selected) < best_size:
                                    best_size = len(selected)
                                    best_list = list(selected)
                            found = True
                            break
                    if found:
                        break
    
    # Final validation and completion
    cov = build_cov(selected)
    if cov.min() < 1:
        uncovered = cov < 1
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for c in range(total):
                cnt = np.count_nonzero(uncovered[balls_arr[c]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = c
            if best_idx < 0:
                break
            selected.append(best_idx)
            uncovered[balls_arr[best_idx]] = False
            if time.time() > start + _DEADLINE - _MARGIN:
                break
    
    # Deduplicate and convert to digit lists
    seen = set()
    final = []
    for c in selected:
        if c in seen:
            continue
        seen.add(c)
        final.append(digits_of(c))
    
    if not final:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return final
