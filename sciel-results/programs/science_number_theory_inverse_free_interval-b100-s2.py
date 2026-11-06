"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses: inv[i] = i^{-1} mod 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # For each start position, find the maximum valid interval length
    best_start = 1
    best_len = 0
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Try to extend the interval from s as far as possible
        # An interval [s, e] is valid if for all x,y in [s,e], x*y % MOD != 1
        # This means: no x in [s,e] has inv[x] also in [s,e], and no x in [s,e] is self-inverse (x=1 or x=1008)
        
        # First check: s itself must not be self-inverse
        if inv[s] == s:
            continue
        
        # Build the set of elements in the growing interval
        # For efficiency, track which inverses are "blocked"
        # An interval [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e]
        
        # Greedy extension: start with e=s, try to extend
        e = s
        valid = True
        while e < N:
            # Check if adding e+1 keeps the interval valid
            next_e = e + 1
            if inv[next_e] == next_e:
                # Self-inverse element cannot be in the interval
                break
            # Check if inv[next_e] is already in [s, e]
            if s <= inv[next_e] <= e:
                break
            # Also need to check: for all existing x in [s,e], is inv[x] == next_e?
            # i.e., is there any x in [s,e] such that inv[x] == next_e?
            # That means x == inv[next_e], which we already checked above
            # Wait, we need to check BOTH directions:
            # - inv[next_e] not in [s, e] (checked)
            # - For all x in [s, e], inv[x] != next_e
            # The second condition: inv[x] == next_e iff x == inv[next_e]
            # So it's the same check! inv[next_e] not in [s, e] covers both.
            
            e = next_e
        
        # Now check the full interval [s, e] is valid
        # We need to verify all pairs, but the greedy construction should ensure validity
        # Let's verify: for all x in [s, e], inv[x] is not in [s, e]
        interval_valid = True
        for x in range(s, e + 1):
            if inv[x] == x:  # self-inverse
                interval_valid = False
                break
            if s <= inv[x] <= e:
                interval_valid = False
                break
        
        if interval_valid and (e - s + 1) > best_len:
            best_len = e - s + 1
            best_start = s
    
    if best_len == 0:
        # Fallback: should not happen, but just in case
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
