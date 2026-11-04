import math
import itertools

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Base-expansion hierarchy approach:
    # U consists of numbers u = sum_{j=0}^{k} b_j * B^j where b_j in {0, 1}
    # and we consider all such combinations up to digit position k.
    # This creates a set where sumset collisions are limited to digit-wise carries,
    # while difference sets expand with the number of active digits.
    
    # We'll search over bases B and number of digits n, and also try to optimize
    # by selecting which digit positions to include (not just consecutive from 0).
    
    for B in range(2, 25):
        for n in range(2, 13):
            # Full set: all numbers with digits 0 or 1 in positions 0 to n-1
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = all_vals
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U[:]
            
            # Try subsets of digit positions: select a subset of positions to use
            if n <= 12:
                for r in range(2, min(n + 1, 8)):  # number of active positions
                    for pos_combo in itertools.combinations(range(n), r):
                        # Generate all combinations of 0/1 on these positions
                        vals = []
                        for mask in range(1 << r):
                            val = 0
                            for idx, pos in enumerate(pos_combo):
                                if mask & (1 << idx):
                                    val += B ** pos
                            vals.append(val)
                        U_sub = vals
                        s_sub = calc_score(U_sub)
                        if s_sub > best_score:
                            best_score = s_sub
                            best = U_sub[:]
    
    return best
