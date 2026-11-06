"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # For each right endpoint R, compute the minimum valid left endpoint L_R
    # such that [L_R, R] contains no modular inverse pair (including self-inverse).
    # We use a backward pass: for each R from 1 to N, we track the constraints
    # imposed by elements in [1, R].

    # For each R, the constraints on L are:
    # 1. L > max of all inv[x] for x in [L, R] where inv[x] < x (since inv[x] must not be in [L, R])
    # 2. L > max of all x for x in [L, R] where inv[x] > x and inv[x] <= R (but this is symmetric)
    # Actually, the constraint is: for every x in [L, R], inv[x] must NOT be in [L, R].
    # This means: for every x in [L, R], either inv[x] < L or inv[x] > R.
    # Since we're fixing R and varying L, the binding constraint is:
    # L > max{inv[x] : x in [L, R], inv[x] < x}  (and similarly for the other direction)
    # But since inv is an involution, if inv[x] < x then x < inv[inv[x]] = x... wait.
    # inv[inv[x]] = x. So if inv[x] < x, then the pair is (inv[x], x) with inv[x] < x.
    # The constraint is that both cannot be in [L, R]. So if x is in [L, R] and inv[x] < x,
    # then we need L > inv[x] (so inv[x] < L, not in interval) OR R < inv[x] (but inv[x] < x <= R, so this is impossible).
    # So the constraint is: L > inv[x] for all x in [L, R] with inv[x] < x.
    # Similarly, if inv[x] > x, then the pair is (x, inv[x]) with x < inv[x].
    # If x is in [L, R] and inv[x] > x, then we need R < inv[x] (impossible since inv[x] > x and we're checking) OR L > x... no.
    # Wait, if x is in [L, R] and inv[x] > x, then inv[x] might be > R, in which case it's fine.
    # The only problem is if inv[x] is also in [L, R]. Since inv[x] > x >= L, we need inv[x] > R.
    # So the constraint is: for all x in [L, R], if inv[x] > x then inv[x] > R, i.e., inv[x] not in [L, R].
    # This is automatically satisfied if inv[x] > R. If inv[x] <= R, then we have a conflict.
    # So: for all x in [L, R], if x < inv[x] <= R, then conflict.
    # Equivalently: for all x in [L, R] with inv[x] > x, we need inv[x] > R.
    # This means: max{inv[x] : x in [L, R], inv[x] > x} > R, i.e., no such x exists with inv[x] <= R.
    # Hmm, this is getting complicated. Let me use the two-pointer approach but more carefully.

    # Two-pointer: maintain L such that [L, R] is valid.
    # For each R, we need L to be large enough that no inverse pair is fully contained in [L, R].
    # The constraints are:
    # - For self-inverse elements (1, 1008): they cannot be in [L, R].
    # - For each x in [L, R], inv[x] must not be in [L, R].

    # Let's define for each R, the minimum L such that [L, R] is valid.
    # We'll compute this using a sweep.

    # For each position p, define conflict_left[p] = the smallest L such that p and inv[p] are both >= L
    # and both <= R for some R. Actually, let's think differently.

    # For a fixed R, the valid L must satisfy:
    # L > max{inv[x] : x in [1, R], inv[x] < x, and x >= L}  -- this is circular.

    # Better: use the fact that the constraint for a pair (a, b) with a < b, a*b ≡ 1 (mod 1009)
    # is that we cannot have both a and b in [L, R]. So L > a or R < b.
    # For fixed R, the constraint is L > a for all pairs (a, b) with b <= R and a < b.
    # So L > max{a : (a, b) is an inverse pair, a < b, b <= R}.
    # Also, self-inverse: L > a for a in {1, 1008} with a <= R.

    # Let's precompute for each R the maximum "left element" of any inverse pair (a, b) with a < b <= R.
    # Plus self-inverse elements.

    # Build list of inverse pairs (a, b) with a < b, a*b ≡ 1 mod 1009.
    # And self-inverse elements.

    # For each R, define max_left[R] = max of:
    #   - a for each inverse pair (a, b) with a < b and b <= R
    #   - a for each self-inverse a with a <= R
    # Then the minimum valid L for right endpoint R is max_left[R] + 1.
    # But we also need L >= 1, and the interval [L, R] must be within [1, 1008].

    # Wait, this isn't quite right. The constraint is that for each pair (a, b) with a < b,
    # we need L > a OR R < b. For fixed R, if b <= R, we need L > a.
    # So L > max{a : (a, b) inverse pair, a < b, b <= R} and L > max{a : a self-inverse, a <= R}.
    # So L_min(R) = 1 + max( max{a : (a,b) pair, b <= R}, max{a : self-inverse, a <= R} ).
    # But we also need L_min(R) <= R for the interval to be non-empty.

    # Let's compute this.

    # First, find all inverse pairs and self-inverse elements.
    self_inv = set()
    pairs = []  # list of (a, b) with a < b, a*b ≡ 1 mod 1009

    for x in range(1, N + 1):
        ix = inv[x]
        if ix == x:
            self_inv.add(x)
        elif ix > x:
            pairs.append((x, ix))

    # For each R, compute max_left[R]
    # We'll sweep R from 1 to N, maintaining the max a for pairs with b <= R, and max self-inv with a <= R.
    max_left = [0] * (N + 1)
    current_max_pair_a = 0
    current_max_self = 0

    # Sort pairs by b
    pairs_sorted = sorted(pairs, key=lambda p: p[1])
    pair_idx = 0
    self_inv_sorted = sorted(self_inv)
    self_idx = 0

    for R in range(1, N + 1):
        # Update with pairs where b == R
        while pair_idx < len(pairs_sorted) and pairs_sorted[pair_idx][1] <= R:
            a, b = pairs_sorted[pair_idx]
            if a > current_max_pair_a:
                current_max_pair_a = a
            pair_idx += 1

        # Update with self-inverse where a == R
        while self_idx < len(self_inv_sorted) and self_inv_sorted[self_idx] <= R:
            a = self_inv_sorted[self_idx]
            if a > current_max_self:
                current_max_self = a
            self_idx += 1

        max_left[R] = max(current_max_pair_a, current_max_self)

    # Now, for each R, L_min = max_left[R] + 1.
    # The interval [L_min, R] is valid if L_min <= R.
    # Length = R - L_min + 1 = R - max_left[R].
    # We want to maximize this over R.

    best_len = 0
    best_start = 1

    for R in range(1, N + 1):
        L_min = max_left[R] + 1
        if L_min <= R:
            length = R - L_min + 1
            if length > best_len:
                best_len = length
                best_start = L_min

    if best_len == 0:
        best_start = 2
        best_len = 1

    # Verify
    def is_valid(s, ln):
        e = s + ln - 1
        if s < 1 or e > N:
            return False
        for x in range(s, e + 1):
            if (x * x) % MOD == 1:
                return False
            ix = inv[x]
            if s <= ix <= e and ix != x:
                return False
        return True

    if not is_valid(best_start, best_len):
        # Fallback
        best_start = 2
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
