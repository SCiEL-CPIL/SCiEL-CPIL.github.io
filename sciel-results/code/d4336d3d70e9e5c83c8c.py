"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008

    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Precompute modular inverses: inv[x] = x^(-1) mod 1009 for x in [1, 1008]
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse residues: x such that x*x % MOD == 1
    # For prime p=1009, these are x=1 and x=p-1=1008
    self_inverse = {1, 1008}

    best_start = 2
    best_len = 1

    # Two-pointer sliding window (O(N) total):
    # For each left pointer S, extend right pointer E as far as possible
    # such that [S, E] is valid (no x*y % MOD == 1 for any x,y in [S,E]).
    #
    # Key: when we extend E to new_E, the only new conflicts are pairs
    # (new_E, x) where x is already in [S, E-1] and new_E * x % MOD == 1,
    # i.e., x == inv[new_E]. So we just check if inv[new_E] is in [S, E-1].
    #
    # Also, if new_E is self-inverse (1 or 1008), we cannot include it.
    #
    # The window is monotone: E never decreases as S increases.

    E = 0  # right pointer, starts before any valid position

    for S in range(1, MOD):  # S from 1 to 1008
        if time.time() > deadline:
            break

        # If S is self-inverse, skip (single-element interval with x*x%MOD==1)
        if S in self_inverse:
            # E must be at least S-1 to maintain invariant
            if E < S - 1:
                E = S - 1
            continue

        # Ensure E >= S - 1 (window must be able to contain S)
        if E < S - 1:
            E = S - 1

        # If E < S, we need to include S itself; check if S is valid
        # S is not self-inverse (checked above), so [S,S] is valid
        # Extend E as far as possible
        while E + 1 <= 1008:  # E can go up to 1008
            new_E = E + 1

            # Check if new_E is self-inverse
            if new_E in self_inverse:
                break

            # Check if adding new_E creates a conflict with existing elements in [S, E]
            inv_new_E = inv[new_E]

            # Conflict if inv_new_E is in [S, E]
            if S <= inv_new_E <= E:
                break

            # No conflict, extend
            E = new_E

        if E >= S:
            current_len = E - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S

    # Verification pass: confirm the best interval is truly valid
    if time.time() < deadline - 5:
        S = best_start
        E_val = best_start + best_len - 1
        valid = True
        for x in range(S, E_val + 1):
            for y in range(x, E_val + 1):
                if x * y % MOD == 1:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Fallback: brute-force search for longest valid interval
            best_start = 2
            best_len = 1
            for S in range(2, MOD):
                if time.time() > deadline:
                    break
                if S in self_inverse:
                    continue
                E_fb = S - 1
                while E_fb + 1 <= 1008:
                    new_E = E_fb + 1
                    if new_E in self_inverse:
                        break
                    # Check all pairs in [S, new_E] involving new_E
                    conflict = False
                    for x in range(S, E_fb + 1):
                        if x * new_E % MOD == 1:
                            conflict = True
                            break
                    if conflict:
                        break
                    E_fb = new_E
                if E_fb >= S:
                    current_len = E_fb - S + 1
                    if current_len > best_len:
                        best_len = current_len
                        best_start = S

    return [best_start, best_len]
# EVOLVE-BLOCK-END
