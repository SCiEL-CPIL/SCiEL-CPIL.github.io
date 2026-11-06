"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    MOD = 1009
    
    # Precompute modular inverses for all x in [1, 1008]
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        # x * inv[x] ≡ 1 (mod 1009)
        # Use Fermat's little theorem: x^(1007) mod 1009
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse residues: x such that x*x % 1009 == 1
    # These are x=1 and x=1008 (since 1008^2 = (1009-1)^2 = 1 - 2*1009 + 1009^2 ≡ 1)
    self_inv = set()
    for x in range(1, MOD):
        if (x * x) % MOD == 1:
            self_inv.add(x)
    # self_inv should be {1, 1008}
    
    best_start = 2
    best_len = 1
    
    # For each possible start, find the longest valid interval
    for start in range(1, MOD):
        if time.time() > deadline:
            break
        
        # Interval must not contain 1 or 1008
        if start <= 1 <= start + best_len - 1:
            # If start=1, we can't include 1, so skip or handle
            pass
        if start <= 1008 <= start + best_len - 1:
            pass
        
        # Try extending the interval from 'start'
        # The interval is [start, start + length - 1], must be within [1, 1008]
        max_end = MOD - 1  # 1008
        end = start
        
        valid = True
        while end <= max_end:
            # Check if end is a self-inverse (1 or 1008)
            if end in self_inv:
                valid = False
                break
            
            # Check if inv[end] is in [start, end]
            ie = inv[end]
            if start <= ie <= end:
                valid = False
                break
            
            end += 1
        
        if valid:
            # The interval [start, end-1] is valid
            length = end - start
            if length > best_len:
                best_len = length
                best_start = start
        else:
            # The interval [start, end-1] is the longest valid starting at 'start'
            # But we need to verify it's actually valid
            length = end - start
            # Verify: for all x in [start, end-1], inv[x] not in [start, end-1]
            # and no self-inverse in [start, end-1]
            is_valid = True
            for x in range(start, end):
                if x in self_inv:
                    is_valid = False
                    break
                if start <= inv[x] <= end - 1:
                    is_valid = False
                    break
            
            if is_valid and length > best_len:
                best_len = length
                best_start = start
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
