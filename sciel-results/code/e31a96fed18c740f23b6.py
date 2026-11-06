"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each start s, the interval [s, s+L-1] is valid iff:
    # 1. No element in the interval is self-inverse (1 or 1008)
    # 2. No pair x,y in the interval has x*y % MOD == 1
    #    i.e., for no x in the interval is inv[x] also in the interval
    
    # Key insight: for a given start s, we can extend the interval until
    # we hit either a self-inverse element or an element whose inverse
    # is already in the interval.
    
    # Since the interval is contiguous [s, e], we can precompute for each
    # position the "conflict" that would occur.
    
    # Approach: For each start s (from 1 to 1008), extend e as far as possible.
    # But we can be smarter: for a fixed s, the constraint is that for all
    # x in [s, e], inv[x] is NOT in [s, e].
    # This means: for all x in [s, e], either inv[x] < s or inv[x] > e.
    
    # Let's precompute for each pair (x, inv[x]) the constraint they impose.
    # If x and inv[x] are both in [s, e], that's a conflict.
    # So for a given s, the maximum e is limited by:
    #   For each x >= s, if inv[x] >= s, then e must be < inv[x] (if inv[x] > x)
    #   or e must be < x (if inv[x] < x, but then inv[x] < s is required... wait)
    
    # More carefully: for start s, as we extend e from s upward:
    # - When we add element e, we need inv[e] not in [s, e-1] (already added)
    #   and e not in [s, inv[e]-1]... no, that's the same thing.
    # - Actually, when adding e, we need inv[e] < s OR inv[e] > e (not yet added)
    #   Wait, inv[e] could be > e and not yet in the interval, but it might be
    #   added later. So we need to check: is inv[e] in [s, e]? If inv[e] == e,
    #   that's the self-inverse case. If inv[e] < e, it's already been considered.
    #   If inv[e] > e, it hasn't been added yet, but will be if we extend far enough.
    
    # So the correct check when extending to include e:
    # - If e is self-inverse (e==1 or e==1008), stop.
    # - If inv[e] is in [s, e-1], stop (conflict with already-added element).
    # - If inv[e] == e, stop (self-inverse, already handled).
    # - If inv[e] > e, we can add e for now, but we must stop before reaching inv[e].
    
    # So for start s, the maximum end is:
    #   min over all x in [s, e_max] of (inv[x] - 1 if inv[x] > x else infinity)
    #   but also we can't go past 1008, and can't include 1 or 1008.
    
    # Let's just do a direct scan but optimized:
    # For each start s, find the maximum valid end.
    
    best_start = 2
    best_len = 1  # at minimum [2,2] is valid since 2*2=4 != 1 mod 1009
    
    for s in range(2, MOD):  # start from 2 (1 is self-inverse)
        if time.time() > deadline:
            break
        if s == 1008:
            break
        
        # Extend from s
        max_end = s
        for e in range(s, MOD):
            if time.time() > deadline:
                break
            if e == 1008:
                break
            # Check if e conflicts
            ie = inv[e]
            if ie == e:
                break
            if ie >= s and ie < e:
                # ie is already in the interval [s, e-1]
                break
            if ie > e:
                # ie will be in the interval if we extend to ie
                # So max_end can be at most ie - 1
                max_end = min(max_end + 1, ie - 1)
                if max_end < e:
                    break
                max_end = e
            else:
                # ie < s, no conflict
                max_end = e
        
        L = max_end - s + 1
        if L > best_len:
            best_len = L
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
