"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements: x^2 ≡ 1 mod p → x = 1 or x = p-1 = 1008
    self_inv = {1, 1008}
    
    best_start = 2
    best_length = 1
    
    # For each start, find the longest valid interval
    for s in range(1, n + 1):
        if s in self_inv:
            continue
        # The interval [s, e] must not contain any x where inv[x] is also in [s, e]
        # and must not contain 1 or 1008
        # Start with e = s, extend as far as possible
        e = s
        # Check if s itself is valid (not self-inverse, already checked)
        # We need to check that no x in [s, e] has inv[x] in [s, e]
        
        # Efficient approach: for each x, the constraint is that if x is in interval,
        # inv[x] must not be in interval. Since interval is [s, e], this means:
        # for all x in [s, e], inv[x] < s or inv[x] > e
        
        # Let's extend e step by step
        valid = True
        for e in range(s, n + 1):
            if e in self_inv:
                valid = False
                break
            # Check if adding e creates a conflict
            # e conflicts if inv[e] is in [s, e]
            if s <= inv[e] <= e:
                valid = False
                break
            # Also check if any x in [s, e-1] now conflicts with e
            # i.e., inv[x] == e for some x in [s, e-1]
            # This is equivalent to: x == inv[e] and x in [s, e-1]
            # But we already check inv[e] in [s, e], so if inv[e] < e and inv[e] >= s, 
            # it would have been caught. Actually inv[e] could be < s which is fine.
            # Wait, we need to check BOTH directions. If x is in [s,e], we need inv[x] not in [s,e].
            # When we add e, we need to check:
            # 1. inv[e] not in [s, e] (covers the case where e's inverse is in interval)
            # 2. For all x in [s, e-1], inv[x] != e (but this is same as x != inv[e], 
            #    and if inv[e] is in [s, e-1], then inv[inv[e]] = e would be in [s,e], 
            #    which is case 1 with x=inv[e])
            # So checking inv[e] not in [s,e] is sufficient!
            if valid:
                if e - s + 1 > best_length:
                    best_length = e - s + 1
                    best_start = s
        # Continue to next start
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
