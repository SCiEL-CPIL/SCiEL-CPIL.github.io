import time
import random
import math

N = 256
# We consider integers 2..256
# Map value to index 0..254
# val = idx + 2

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

# Precompute primes
PRIMES = [p for p in range(2, N + 1) if is_prime(p)]

# We will use indices 0 to 254 representing values 2 to 256
# idx = val - 2

# Precompute conflict masks for antichain (divisibility)
# div_mask[i] has bit j set if val_i divides val_j
M = N - 1  # 255 elements
VALS = list(range(2, N + 1))

div_mask = [0] * M
for i in range(M):
    vi = VALS[i]
    mask = 0
    for j in range(M):
        vj = VALS[j]
        if vj % vi == 0:
            mask |= (1 << j)
    div_mask[i] = mask

# Precompute "i divides j * k" condition
# For a fixed i and j, we can determine which k's are invalid.
# i | j * k  <=>  i / gcd(i, j) | k
# Let g = gcd(i, j). Then i/g | k.
# So k must be a multiple of i/g.
# We can precompute a mask for each pair (i, j) of k's that are multiples of i/g.
# But M is 255, so M^2 masks of size M bits is 255*255*32 bytes ~ 2MB. Feasible.

# mult_mask[i][j] will have bit k set if VALS[i] divides VALS[j] * VALS[k]
# Note: This is not symmetric.
# We need to check for any distinct a,b,c in S:
# a | b*c
# b | a*c
# c | a*b

# To check if we can add x to S:
# 1. Antichain: x not divide any y in S, and no y in S divides x.
#    (x in div_mask[y] or y in div_mask[x])
# 2. Triple: For all distinct a,b in S:
#    a | x * b  (x plays role of c? No, a,b,x are the triple)
#    x | a * b
#    b | a * x
#    
# Let's define conflict masks for adding x to current set S (bitmask).
# Condition 1: Antichain with S
# If (div_mask[x] & S) != 0: fail (x divides some y in S)
# If (div_mask_inv[x] & S) != 0: fail (some y in S divides x) -- need div_mask_inv or just check y|x.
# Actually, div_mask[i] has j if i|j.
# So if div_mask[x] & S != 0, then x|y for some y in S.
# If div_mask[y] & BIT[x] != 0 for some y in S, then y|x.
# We can precompute div_mask_inv or just iterate. 
# Better: Precompute `divides_mask[i]` (i divides j) and `is_divided_by_mask[i]` (j divides i).
# is_divided_by_mask[i] = OR of BIT[j] for all j such that j|i.

is_divided_by_mask = [0] * M
for i in range(M):
    vi = VALS[i]
    mask = 0
    for j in range(M):
        vj = VALS[j]
        if vi % vj == 0:
            mask |= (1 << j)
    is_divided_by_mask[i] = mask

# Condition 2: Triple constraints with pairs in S
# For each pair (a, b) in S (distinct), we need:
# 1. a | x * b  =>  a / gcd(a, x) | b  (Wait, a | x*b. Let g = gcd(a,x). a/g | b. So b must NOT be a multiple of a/g? No, a | x*b is the BAD condition. We want a NOT divide x*b.)
#    So we need: a/g | b is FALSE.
#    Equivalently: b is NOT a multiple of a/g.
# 2. b | x * a  =>  b / gcd(b, x) | a is FALSE.
# 3. x | a * b  =>  x | a * b is FALSE.

# This is hard to precompute as a single mask for x against S.
# Instead, we can iterate over pairs in S. |S| is at most ~100. Pairs ~ 5000.
# For each pair, check 3 conditions. 5000 * 3 = 15000 ops.
# Adding 1 element: 15000 ops.
# Local search: many adds/swaps. Might be slow in Python if not optimized.

# Optimization:
# Precompute `bad_k_mask[i][j]`: mask of k such that VALS[i] divides VALS[j] * VALS[k].
# Then for adding x:
# For each a in S:
#   We need to check if there exists b in S (b != a) such that:
#   - a | x * b  <=> b in bad_k_mask[a][x]
#   - x | a * b  <=> b in bad_k_mask[x][a] (Wait, x | a*b. Here roles are: divisor=x, factors=a,b. So b must be in mask for x,a? No.
#     bad_k_mask[i][j] is k such that i | j * k.
#     So x | a * b means b is in bad_k_mask[x][a].
#   - b | a * x  <=> a is in bad_k_mask[b][x]. This depends on b.

# Let's precompute `bad_k_mask[i][j]` for all i,j.
# Size 255*255*32 bytes = 2MB. OK.

bad_k_mask = [[0] * M for _ in range(M)]
for i in range(M):
    vi = VALS[i]
    for j in range(M):
        vj = VALS[j]
        g = math.gcd(vi, vj)
        target = vi // g
        # k such that vj * vk % vi == 0
        # Equivalent to vk % target == 0
        mask = 0
        for k in range(M):
            vk = VALS[k]
            if vk % target == 0:
                mask |= (1 << k)
        bad_k_mask[i][j] = mask

def can_add(S, x):
    # S is bitmask
    if S & (1 << x):
        return False
    
    # 1. Antichain check
    if (div_mask[x] & S) or (is_divided_by_mask[x] & S):
        return False
    
    # 2. Triple check
    # Iterate over all a in S
    # For each a, we need to ensure no b in S (b != a) causes conflict.
    # Conflicts involving x and a and some b:
    # A) a | x * b  => b in bad_k_mask[a][x]
    # B) x | a * b  => b in bad_k_mask[x][a]
    # C) b | a * x  => a in bad_k_mask[b][x]  (This is a condition on b for fixed a,x? No, it's a condition on a for fixed b,x. 
    #    For a fixed a and x, we need to check if there exists b in S such that b | a * x.
    #    This is equivalent to: a * x is in the set of multiples of b.
    #    Or: b is a divisor of a * x.
    #    Let's precompute `divs_of_prod_mask[i][j]`: mask of k such that VALS[k] divides VALS[i] * VALS[j].
    #    Then C is: a in S? No, b in S.
    #    We need: (divs_of_prod_mask[a][x] & S) == 0 ? 
    #    Wait. If b in S and b | a*x, then conflict.
    #    So we need: divs_of_prod_mask[a][x] & S == 0.
    
    # Let's precompute divs_of_prod_mask
    # This is M^2 masks. 2MB.
    
    pass

# Re-adding precompute for divs_of_prod
divs_of_prod_mask = [[0] * M for _ in range(M)]
for i in range(M):
    vi = VALS[i]
    for j in range(M):
        vj = VALS[j]
        prod = vi * vj
        mask = 0
        for k in range(M):
            vk = VALS[k]
            if prod % vk == 0:
                mask |= (1 << k)
        divs_of_prod_mask[i][j] = mask

def can_add(S, x):
    if S & (1 << x):
        return False
    
    # Antichain
    if (div_mask[x] & S) or (is_divided_by_mask[x] & S):
        return False
    
    # Triple
    # Iterate a in S
    S_temp = S
    while S_temp:
        lsb = S_temp & -S_temp
        a = lsb.bit_length() - 1
        S_temp ^= lsb
        
        # A) a | x * b. Bad if exists b in S such that b in bad_k_mask[a][x].
        # Note: b must be distinct from a and x. x is not in S. b is in S.
        # If a is in bad_k_mask[a][x], it means a | x*a. Always true. But b must be distinct from a.
        # So we check (bad_k_mask[a][x] & S & ~(1 << a)).
        if (bad_k_mask[a][x] & S & ~(1 << a)):
            return False
            
        # B) x | a * b. Bad if exists b in S such that b in bad_k_mask[x][a].
        # b distinct from a.
        if (bad_k_mask[x][a] & S & ~(1 << a)):
            return False
            
        # C) b | a * x. Bad if exists b in S such that b in divs_of_prod_mask[a][x].
        # b distinct from a.
        if (divs_of_prod_mask[a][x] & S & ~(1 << a)):
            return False
            
    return True

def construct():
    deadline = time.time() + 50.0
    best_S = 0
    best_size = 0
    
    # Initialize with all numbers > 128
    S = 0
    for i in range(M):
        if VALS[i] > 128:
            S |= (1 << i)
            
    # Clean up S to ensure validity
    # Since we added greedily, it might be invalid.
    # Better to start empty and add?
    # Or start with >128 and remove conflicts.
    
    # Let's do a greedy construction from scratch with randomization
    # Start with large primes (always safe)
    base_S = 0
    for i in range(M):
        if VALS[i] in PRIMES and VALS[i] > 128:
            base_S |= (1 << i)
            
    best_S = base_S
    best_size = bin(base_S).count('1')
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        S = base_S
        
        # Try to add composites > 128
        cands_large = [i for i in range(M) if VALS[i] > 128 and not (S & (1 << i))]
        rng.shuffle(cands_large)
        for i in cands_large:
            if can_add(S, i):
                S |= (1 << i)
                
        # Try to add small numbers
        cands_small = [i for i in range(M) if VALS[i] <= 128 and not (S & (1 << i))]
        rng.shuffle(cands_small)
        for i in cands_small:
            if can_add(S, i):
                S |= (1 << i)
                
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_S = S
            
        # Local search: swap
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Pick a random element to remove
            mems = [i for i in range(M) if S & (1 << i)]
            if not mems:
                break
            y = rng.choice(mems)
            S1 = S & ~(1 << y)
            
            # Try to add a new element
            cands = [i for i in range(M) if not (S1 & (1 << i))]
            rng.shuffle(cands)
            improved = False
            for x in cands:
                if can_add(S1, x):
                    S2 = S1 | (1 << x)
                    # Greedy add more
                    changed = True
                    while changed:
                        changed = False
                        for z in cands: # Reuse shuffled list? Better to shuffle again or use sorted
                            if not (S2 & (1 << z)) and can_add(S2, z):
                                S2 |= (1 << z)
                                changed = True
                                break
                    if bin(S2).count('1') > bin(S).count('1'):
                        S = S2
                        improved = True
                        break
            if not improved:
                # Try to add without removing?
                # (Already done in initial phase, but maybe after swaps?)
                pass
                
    # Convert back to list of ints
    res = []
    for i in range(M):
        if best_S & (1 << i):
            res.append(VALS[i])
            
    return res

construct()
