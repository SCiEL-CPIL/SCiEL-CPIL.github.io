"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    start_time = time.time()
    deadline = start_time + 80

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse residues
    self_inverse = {1, 1008}
    
    # Build all inverse pairs (a, b) with a < b, a * b % MOD == 1
    # Exclude self-inverse pairs
    pairs = []
    seen = set()
    for x in range(2, 1008):  # skip 1 and 1008
        if x in self_inverse:
            continue
        y = inv[x]
        if y == x:
            continue
        if y < 2 or y > 1007:
            continue
        a, b = min(x, y), max(x, y)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    
    # For each right endpoint E (2 to 1007), compute:
    # max_a[E] = max{a : (a,b) is a pair, b <= E}
    # If no such pair, max_a[E] = 1 (so S must be >= 2)
    max_a = [1] * (MOD + 1)
    for a, b in pairs:
        if b > max_a[b] if b < len(max_a) else False:
            pass
    # More carefully: for each E, max_a[E] = max of all a where b <= E
    # Initialize to 1 (meaning S must be at least 2)
    max_a_arr = [1] * (MOD + 1)
    for a, b in pairs:
        if a > max_a_arr[b]:
            max_a_arr[b] = a
    
    # Now propagate: max_a_arr[E] = max(max_a_arr[E], max_a_arr[E-1])
    # Because if a pair has b <= E-1, it also has b <= E
    for E in range(2, MOD):
        if max_a_arr[E-1] > max_a_arr[E]:
            max_a_arr[E] = max_a_arr[E-1]
    
    # For each E in [2, 1007], the minimum valid S is max_a_arr[E] + 1
    # But S must also be >= 2
    # The interval [S, E] has length E - S + 1
    # To maximize length for given E, use smallest valid S: S = max(2, max_a_arr[E] + 1)
    # Length = E - S + 1 = E - max(2, max_a_arr[E] + 1) + 1
    
    best_start = 2
    best_len = 1
    
    for E in range(2, 1008):
        if time.time() > deadline:
            break
        S_min = max(2, max_a_arr[E] + 1)
        if S_min <= E:
            current_len = E - S_min + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S_min
    
    # Also check single-element intervals at positions not self-inverse
    # Already covered since S_min <= E gives at least length 1 when E >= 2
    
    # Verify the result
    S = best_start
    E_val = best_start + best_len - 1
    valid = True
    for x in range(S, E_val + 1):
        if x in self_inverse:
            valid = False
            break
        y_inv = inv[x]
        if S <= y_inv <= E_val:
            valid = False
            break
    if not valid:
        # Fallback: brute force for small intervals
        best_start = 2
        best_len = 1
        for S in range(2, 1008):
            for E in range(S, 1008):
                ok = True
                for x in range(S, E + 1):
                    if x in self_inverse:
                        ok = False
                        break
                    if S <= inv[x] <= E:
                        ok = False
                        break
                if ok:
                    if E - S + 1 > best_len:
                        best_len = E - S + 1
                        best_start = S
                else:
                    break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
