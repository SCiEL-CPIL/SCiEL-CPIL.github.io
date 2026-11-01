import itertools
import math

def score_set(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    s_size = len(sum_set)
    d_size = len(diff_set)
    if s_size == 0:
        return 0
    ratio = d_size / s_size
    if ratio <= 1:
        return 1.0 # Or 1 + negative? Log of <1 is negative.
        # The formula is 1 + log(R)/log(2M+1). If R<1, score < 1.
        # We want to maximize.
    val = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

best_val = 0
best_set = [0]

# Search up to max U = 15 or so, and size up to 6
max_u_limit = 20
max_size = 6

# We must include 0.
# Iterate over possible max_u
for max_u in range(1, max_u_limit + 1):
    # Candidates are subsets of {0, ..., max_u} containing 0 and max_u
    # To keep it fast, we can just check all subsets of {1, ..., max_u-1}
    # But 2^19 is too big.
    # Let's restrict size.
    # Actually, for fixed max_u, we want to choose intermediate elements to maximize score.
    
    # Let's just iterate over combinations of size 2 to 5 from {1, ..., max_u-1}
    # Plus 0 and max_u.
    
    if max_u == 1:
        U = [0, 1]
        v = score_set(U)
        if v > best_val:
            best_val = v
            best_set = U
        continue
        
    for size in range(2, max_size + 1):
        # We need 'size-2' elements from 1 to max_u-1
        if max_u - 1 < size - 2:
            continue
        for combo in itertools.combinations(range(1, max_u), size - 2):
            U = [0] + list(combo) + [max_u]
            v = score_set(U)
            if v > best_val:
                best_val = v
                best_set = U

print(f"Best Set: {best_set}")
print(f"Best Score: {best_val}")
