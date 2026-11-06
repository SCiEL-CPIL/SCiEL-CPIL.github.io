import time

# EVOLVE-BLOCK-START

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    deadline = time.time() + 80  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 1
    best_len = 0

    # O(N) backward sweep:
    # For each start s, the maximal valid end e(s) is the largest e >= s
    # such that for all x in [s, e], inv[x] is not in [s, e].
    #
    # Key observation: as s decreases, the constraint set shrinks.
    # We can maintain a running "right boundary" that tracks the tightest
    # upper bound on e imposed by elements at or after s.
    #
    # For a fixed s, an element x in [s, e] conflicts if inv[x] >= s and
    # inv[x] <= e. So e must be < inv[x] for any x with inv[x] >= s.
    #
    # Backward approach: process s from N down to 1.
    # Maintain an array or structure that for each potential x >= s,
    # tells us the constraint. Actually, let's think differently.
    #
    # For each x, the constraint is: if x is in the interval [s,e], then
    # inv[x] must NOT be in [s,e]. This means either inv[x] < s or inv[x] > e.
    # Since we're looking for the max e given s, and inv[x] is fixed:
    #   - If inv[x] < s: no constraint on e from this x.
    #   - If inv[x] >= s: we need e < inv[x], so e <= inv[x] - 1.
    #
    # So for start s: e_max(s) = min over x in [s, e_max] of (inv[x]-1 if inv[x]>=s else N)
    # But this is self-referential. Let's use a different approach.
    #
    # Alternative: For each s, the valid interval [s, e] requires that
    # no pair (x, inv[x]) both fall in [s, e]. This is equivalent to:
    # the set [s, e] contains no element and its inverse simultaneously.
    #
    # Let's define: for each s, what is the largest e such that
    # {x in [s,e] : inv[x] in [s,e]} is empty?
    #
    # We can precompute for each position x, the "conflict partner" inv[x].
    # An interval [s,e] is valid iff for all x in [s,e], inv[x] < s or inv[x] > e.
    #
    # O(N) method: For each s from N down to 1, maintain the minimum
    # of (inv[x] - 1) over all x >= s where inv[x] >= s. But s changes,
    # so the condition inv[x] >= s changes too.
    #
    # Let me just do a smarter O(N log N) or careful O(N) pass.
    #
    # Actually, the original O(N^2) is fine for N=1008 (about 500K ops).
    # But let's optimize the inner loop.

    # For each start s, find max valid end e
    for s in range(1, MOD):
        if time.time() > deadline:
            break

        if inv[s] == s:
            continue

        limit = N
        max_valid_e = s - 1

        for x in range(s, MOD):
            ix = inv[x]
            if ix >= s:
                if ix - 1 < limit:
                    limit = ix - 1
            if x > limit:
                break
            max_valid_e = x

        length = max_valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

    # Fallback
    if best_len == 0:
        for s in range(1, MOD):
            if inv[s] != s:
                best_start = s
                best_len = 1
                break
    if best_len == 0:
        best_start = 1
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
