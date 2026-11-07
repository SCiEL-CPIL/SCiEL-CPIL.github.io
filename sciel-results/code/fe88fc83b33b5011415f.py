"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

# LCM of the divisors
def _lcm(a, b):
    return a * b // math.gcd(a, b)

LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)
# LCM = 30030

# Precompute for each divisor d, the set of residues r in [0, LCM) such that
# a multiple of d exists at position start + offset where offset corresponds to residue r.
# Actually, we want: given a window [W, W+R], for each divisor d_i, the available multiples
# are k*d_i such that W <= k*d_i <= W+R.
# We can think of it as: for a fixed window start W, the multiples of d_i in [W, W+R]
# are those k where ceil(W/d_i) <= k <= floor((W+R)/d_i).
# Two multiples (one for d_i, one for d_j) collide if they are equal.
# Since we want DISTINCT multiples, we need to assign to each divisor d_i a distinct multiple
# within [W, W+R].

# The strategy: 
# For a given window size R, check if there exists a start W such that we can assign
# distinct multiples. 
# 
# Key insight: The pattern of available multiples repeats every LCM. So we only need to
# check W mod LCM. For each residue r in [0, LCM), we can check feasibility.
#
# For a window [W, W+R], the count of multiples of d_i in the window is either
# floor(R/d_i) or floor(R/d_i)+1.
#
# A simpler approach: For small R, we can try all possible windows.
# But R could be large. Let's think about the minimum possible R.
#
# The minimum R is at least max(d_i) - 1 = 39 - 1 = 38? No, that's not right.
# The minimum R is determined by the need to fit 10 distinct multiples.
#
# Let's try a different approach: 
# 1. For a given R, for each residue r in [0, LCM), determine if the window [r, r+R]
#    (mod LCM, but we need to be careful about wrap-around) contains enough distinct
#    multiples.
#
# Actually, since LCM = 30030 and R might be small, we can just iterate over all
# possible start positions in [0, LCM) and for each, check if we can find 10 distinct
# multiples.
#
# For a window [W, W+R] where W is in [0, LCM), the multiples of d_i in the window are:
# k_i_min = ceil(W / d_i)
# k_i_max = floor((W+R) / d_i)
# The available multiples are {k_i_min * d_i, ..., k_i_max * d_i}.
#
# We need to select one from each set such that all are distinct.
# This is a bipartite matching problem, but since the sets are intervals of multiples,
# we can use a greedy approach or check Hall's condition.
#
# Actually, since the multiples are just integers, and we need them to be distinct,
# we can model this as: can we assign distinct values from the union of all available
# multiples, one per divisor?
#
# A simpler check: the total number of distinct multiples available must be at least 10,
# and no single value can be the only option for multiple divisors in a way that causes
# conflict.
#
# Let's try a more direct approach:
# For a given R and start W, collect all available multiples for each divisor.
# Then check if there's a system of distinct representatives (SDR).
# We can use the Hungarian algorithm or a simple backtracking since N=10 is small.
#
# But we need to do this for many W values. Let's optimize:
# For a fixed R, iterate W from 0 to LCM-1. For each W, compute the available multiples
# for each divisor. Then check SDR.
#
# To speed up, we can precompute for each divisor d_i and each possible W mod d_i,
# the range of k values. But this might not help much.
#
# Alternative: Since N=10 is small, we can use a bitmask approach.
# For each W, the available multiples for d_i form a contiguous range of multiples.
# We can represent the "occupied" values as a set and try to assign.
#
# Let's try: for each W in range(LCM), for each divisor d_i, compute the list of
# available multiples. Then try to find an SDR using backtracking.
#
# But LCM = 30030 and for each W we do 10 divisors * backtracking. This might be slow.
#
# Let's try a different approach: 
# The minimum R is small. Let's binary search on R.
# For a given R, check if there exists W in [0, LCM) such that the window [W, W+R]
# admits an SDR.
#
# To check for a given W and R:
# For each divisor d_i, the available multiples are:
#   k_min = ceil(W / d_i)
#   k_max = floor((W + R) / d_i)
#   multiples = [k_min * d_i, (k_min+1) * d_i, ..., k_max * d_i]
#
# We need to pick one from each list such that all picks are distinct.
#
# Since the lists are small (at most R/d_i + 1 elements), we can use a simple
# backtracking or even a greedy approach with sorting.
#
# Actually, a key observation: the multiples for different divisors are interleaved.
# The only way two divisors can have the same multiple is if that multiple is a
# common multiple of both divisors.
#
# Let's just implement the backtracking check for SDR. Since N=10 and each list
# has at most ~R/d_i + 1 elements, and R is small (say < 100), this should be fast.

def _has_sdr(avail):
    """Check if there's a system of distinct representatives.
    avail[i] is a list of available values for divisor i.
    Returns True if SDR exists."""
    n = len(avail)
    # Sort divisors by the size of their available list (most constrained first)
    order = sorted(range(n), key=lambda i: len(avail[i]))
    
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for v in avail[i]:
            if v not in used:
                used.add(v)
                if backtrack(idx + 1):
                    return True
                used.remove(v)
        return False
    
    return backtrack(0)

def _check_R(R):
    """Check if there exists a window of size R with an SDR."""
    for W in range(LCM):
        avail = []
        feasible = True
        for d in DIVISORS:
            k_min = (W + d - 1) // d  # ceil(W/d)
            k_max = (W + R) // d       # floor((W+R)/d)
            if k_min > k_max:
                feasible = False
                break
            # The multiples are k_min*d, (k_min+1)*d, ..., k_max*d
            # We store them as a list
            avail.append([k * d for k in range(k_min, k_max + 1)])
        if not feasible:
            continue
        if _has_sdr(avail):
            return True, W, avail
    return False, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Try R from 0 upwards
    # The minimum R is at least 9 (since we need 10 distinct values)
    # But actually, the minimum R is determined by the divisors.
    # Let's start from R=0 and go up.
    
    best_R = None
    best_W = None
    best_avail = None
    
    for R in range(0, 500):
        if time.time() > deadline:
            break
        found, W, avail = _check_R(R)
        if found:
            best_R = R
            best_W = W
            best_avail = avail
            break
    
    if best_R is None:
        # Fallback: just return the divisors themselves (which is valid but not optimal)
        return DIVISORS[:]
    
    # Now extract the actual values
    # We need to re-run the SDR assignment to get the actual values
    # Since we know W and R, we can recompute
    W = best_W
    R = best_R
    
    avail = []
    for d in DIVISORS:
        k_min = (W + d - 1) // d
        k_max = (W + R) // d
        avail.append([k * d for k in range(k_min, k_max + 1)])
    
    # Find the SDR
    n = len(avail)
    order = sorted(range(n), key=lambda i: len(avail[i]))
    used = set()
    assignment = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for v in avail[i]:
            if v not in used:
                used.add(v)
                assignment[i] = v
                if backtrack(idx + 1):
                    return True
                used.remove(v)
                assignment[i] = None
        return False
    
    if not backtrack(0):
        # Should not happen
        return DIVISORS[:]
    
    result = []
    for i in range(n):
        val = assignment[i]
        if val is None or val < 1 or val > MAX_VAL:
            return DIVISORS[:]
        result.append(int(val))
    
    return result
# EVOLVE-BLOCK-END
