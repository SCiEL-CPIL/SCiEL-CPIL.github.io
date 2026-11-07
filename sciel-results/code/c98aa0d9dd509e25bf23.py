import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for the 90s wall-clock limit

    N = 12
    period = 27720  # lcm(1..12)
    max_L = 150

    # Precompute for each i (1..12), each residue r (0..i-1), and each L (0..max_L)
    # a bitmask of positions j in [0, L) where (m + 1 + j) % i == r
    # Actually, we need: for a given m, the residue of (m+1) mod i determines which positions work.
    # Let's reframe: for i in 1..12, and a window (m, m+L], position j (0-indexed from m+1)
    # works for i if (m + 1 + j) % i == 0.
    # So the "residue" we care about is m % i.
    # If m % i == r, then (m + 1 + j) % i == (r + 1 + j) % i.
    # We need (r + 1 + j) % i == 0, i.e., j % i == (i - r - 1) % i.

    # precomputed[i][r][L] = bitmask of j in [0, L) such that (m + 1 + j) % i == 0
    # where r = m % i
    # This means j % i == (i - 1 - r) % i

    precomputed = {}
    for i in range(1, N + 1):
        precomputed[i] = {}
        for r in range(i):
            needed_j_mod = (i - 1 - r) % i
            masks_for_L = [0] * (max_L + 1)
            cur_mask = 0
            for L in range(1, max_L + 1):
                j = L - 1
                if j % i == needed_j_mod:
                    cur_mask |= (1 << j)
                masks_for_L[L] = cur_mask
            precomputed[i][r] = masks_for_L

    def compute_f(m):
        # Compute m mod i for each i
        residues = [0] * (N + 1)
        for i in range(1, N + 1):
            residues[i] = m % i

        for L in range(1, max_L + 1):
            masks = [0] * (N + 1)
            union_mask = 0
            valid = True
            for i in range(1, N + 1):
                mask = precomputed[i][residues[i]][L]
                if mask == 0:
                    valid = False
                    break
                masks[i] = mask
                union_mask |= mask

            if not valid:
                continue

            if union_mask.bit_count() < N:
                continue

            # Sort indices by mask popcount (most constrained first)
            indices = list(range(1, N + 1))
            indices.sort(key=lambda i: masks[i].bit_count())

            # Backtracking with memoization on (idx, used_mask)
            # used_mask is a bitmask of positions already assigned
            memo = {}
            
            def backtrack(idx, used):
                if idx == N:
                    return True
                key = (idx, used)
                if key in memo:
                    return memo[key]
                
                i = indices[idx]
                avail = masks[i] & ~used
                while avail:
                    bit = avail & -avail
                    avail ^= bit
                    if backtrack(idx + 1, used | bit):
                        memo[key] = True
                        return True
                
                memo[key] = False
                return False

            if backtrack(0, 0):
                return L

        return max_L

    # Search over m in [0, period)
    best_m = 0
    best_f = 0

    for m in range(period):
        if time.time() > deadline:
            break

        f_val = compute_f(m)

        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)
