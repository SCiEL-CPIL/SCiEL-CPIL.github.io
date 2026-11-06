"""Model-and-optimize: parametrize interval by start s, fit maximal valid end e."""
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 82.0  # explicit safety margin before the 90 s kill

    MOD = 1009  # prime
    N = MOD - 1  # residues 1..1008

    # --- Model construction -------------------------------------------------
    # Modular inverse of each residue (1009 is prime => inv exists for all).
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For an interval [s, e] to be valid we need, for every x in [s, e]:
    #   inv[x] < s   or   inv[x] > e          (i.e. inv[x] NOT in [s, e])
    # This is equivalent to: no x in [s,e] has inv[x] in [s,e].
    # 
    # Key observation: define for each x, conflict_partner = inv[x].
    # An interval [s,e] is valid iff for all x in [s,e], inv[x] is outside [s,e].
    # 
    # For fixed s, max valid e: scan x from s upward, track min_inv_ge_s = 
    #   min{inv[x] : x in [s, current], inv[x] >= s}.
    # The constraint is e < min_inv_ge_s, so max e = min(min_inv_ge_s - 1, current)
    # but we must also ensure current <= min_inv_ge_s - 1.
    # Break when current >= min_inv_ge_s (i.e., min_inv_ge_s <= current).

    INF = 10 ** 9
    best_start = None
    best_len = 0

    # Self-inverse residues: x where inv[x] == x, i.e., x^2 ≡ 1 (mod 1009)
    # These are x=1 and x=1008 (since 1009 is prime, x^2≡1 has exactly 2 solutions)
    self_inverse = set()
    for x in range(1, MOD):
        if inv[x] == x:
            self_inverse.add(x)

    # O(N^2) forward sweep - N=1008 so ~500K operations, trivially fast
    for s in range(1, MOD):
        if time.time() > deadline:
            break
        
        # If s is self-inverse, any interval containing s is invalid (since s*s % MOD = 1)
        if s in self_inverse:
            continue
        
        min_u = INF
        max_valid_e = s - 1
        for x in range(s, MOD):
            ix = inv[x]
            # If inv[x] >= s, then we need e < inv[x], so upper bound is inv[x]-1
            # If inv[x] < s, no upper bound from this x
            if ix >= s:
                if ix - 1 < min_u:
                    min_u = ix - 1
            
            # Check if x itself is still valid: need min_u >= x
            # (i.e., all constraints allow e to be at least x)
            if min_u >= x:
                max_valid_e = x
            else:
                # Once min_u < x, for any x' > x, min_u can only decrease or stay,
                # and x' increases, so min_u < x' always. Break.
                break
        
        length = max_valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

    # Fallback: find any valid singleton
    if best_start is None:
        for s in range(1, MOD):
            if s not in self_inverse:
                best_start = s
                best_len = 1
                break
    if best_start is None:
        best_start = 2
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
