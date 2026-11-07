import time
import random
from itertools import combinations

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

# ---------- Precomputation ----------

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

PRIMES = [p for p in range(2, N + 1) if _is_prime(p)]
PRIME_SET = set(PRIMES)
NPRIMES = len(PRIMES)

# Map each prime to its index
PRIME_IDX = {p: i for i, p in enumerate(PRIMES)}

# For each number 2..256, compute the set of primes it contains (as a bitmask)
def _prime_mask(n):
    mask = 0
    x = n
    for p in PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            mask |= (1 << PRIME_IDX[p])
            while x % p == 0:
                x //= p
    if x > 1:
        mask |= (1 << PRIME_IDX[x])
    return mask

# For composites: the mask of primes that would be "blocked" if this composite is in S
COMPOSITE_CANDIDATES = []
for a in range(2, N + 1):
    if a in PRIME_SET:
        continue
    mask = _prime_mask(a)
    # Only consider composites with exactly 2 distinct prime factors
    if bin(mask).count('1') == 2:
        COMPOSITE_CANDIDATES.append((a, mask))

# Sort by number of primes blocked (ascending), then by value
COMPOSITE_CANDIDATES.sort(key=lambda x: (bin(x[1]).count('1'), x[0]))
NCAND = len(COMPOSITE_CANDIDATES)

# Precompute conflict matrix
# Two composites x, y conflict if |Px \ Py| <= 1 OR |Py \ Px| <= 1
conflict = [[False] * NCAND for _ in range(NCAND)]
for i in range(NCAND):
    mi = COMPOSITE_CANDIDATES[i][1]
    for j in range(i + 1, NCAND):
        mj = COMPOSITE_CANDIDATES[j][1]
        diff_ij = mi & ~mj
        diff_ji = mj & ~mi
        if bin(diff_ij).count('1') <= 1 or bin(diff_ji).count('1') <= 1:
            conflict[i][j] = True
            conflict[j][i] = True

cand_blocks = [mask for (_, mask) in COMPOSITE_CANDIDATES]

# ---------- Branch and Bound ----------

best_size = NPRIMES
best_set = list(PRIMES)
best_included = []

def _upper_bound(included, start_idx):
    """Upper bound on final set size given current included set and start index."""
    count = len(included)
    for i in range(start_idx, NCAND):
        ok = True
        for j in included:
            if conflict[i][j]:
                ok = False
                break
        if ok:
            count += 1
    return count

def _bb(included, start_idx, current_size):
    """Branch and bound search."""
    global best_size, best_set, best_included
    deadline_check = time.time()
    
    if deadline_check > DEADLINE:
        return
    
    # Upper bound
    ub = _upper_bound(included, start_idx)
    if ub <= best_size:
        return
    
    # Try to add candidates
    for i in range(start_idx, NCAND):
        if deadline_check > DEADLINE:
            return
        # Check if candidate i is compatible with all included
        ok = True
        for j in included:
            if conflict[i][j]:
                ok = False
                break
        if not ok:
            continue
        
        # Add candidate i
        included.append(i)
        new_size = current_size + 1
        if new_size > best_size:
            best_size = new_size
            best_included = included[:]
            best_set = list(PRIMES) + [COMPOSITE_CANDIDATES[j][0] for j in included]
        
        # Recurse
        _bb(included, i + 1, new_size)
        
        # Remove candidate i
        included.pop()
        
        if time.time() > DEADLINE:
            return

# ---------- Local Search (swap-based) ----------

def _local_search(initial_included):
    """Try to improve via swaps: remove one, add another."""
    global best_size, best_set, best_included
    included = initial_included[:]
    cur_size = len(included)
    improved = True
    
    while improved and time.time() < DEADLINE - 2:
        improved = False
        for i in range(len(included)):
            if time.time() > DEADLINE - 2:
                break
            # Try removing included[i]
            removed = included[i]
            new_included = [x for x in included if x != removed]
            
            # Try adding any candidate not in new_included
            for j in range(NCAND):
                if j in new_included:
                    continue
                # Check compatibility
                ok = True
                for k in new_included:
                    if conflict[j][k]:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Add j
                new_included.append(j)
                new_size = len(new_included)
                
                if new_size > cur_size:
                    included = new_included
                    cur_size = new_size
                    improved = True
                    break
            if improved:
                break
    
    if cur_size > best_size:
        best_size = cur_size
        best_included = included[:]
        best_set = list(PRIMES) + [COMPOSITE_CANDIDATES[j][0] for j in included]
    
    return included

# ---------- Main ----------

def construct():
    global best_size, best_set, best_included
    
    start_time = time.time()
    
    # Initial: just all primes
    best_size = NPRIMES
    best_set = list(PRIMES)
    best_included = []
    
    # Greedy: add composites in order of least blocking
    greedy_included = []
    for i in range(NCAND):
        if time.time() > DEADLINE - 10:
            break
        ok = True
        for j in greedy_included:
            if conflict[i][j]:
                ok = False
                break
        if ok:
            greedy_included.append(i)
    
    if len(greedy_included) > best_size:
        best_size = len(greedy_included)
        best_included = greedy_included[:]
        best_set = list(PRIMES) + [COMPOSITE_CANDIDATES[j][0] for j in greedy_included]
    
    # Local search on greedy result
    ls_included = _local_search(greedy_included)
    
    # Branch and bound (with time limit)
    if time.time() < DEADLINE - 5:
        _bb([], 0, 0)
    
    # Final local search on best
    if time.time() < DEADLINE - 2:
        _local_search(best_included)
    
    # Verify and return
    result = sorted(best_set)
    # Ensure all are ints in range
    result = [int(x) for x in result if 2 <= x <= 256]
    
    return list(result)
