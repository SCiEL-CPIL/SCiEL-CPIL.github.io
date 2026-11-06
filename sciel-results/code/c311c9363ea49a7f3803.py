"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    p = 1009
    
    # Precompute quadratic residues and their smallest positive root
    # is_res[i] = 1 if i is a non-zero quadratic residue mod p, else 0
    # root_of[i] = smallest x in [1, p-1] such that x^2 % p == i, or 0 if not a residue
    is_res = [0] * (p + 1)  # index 0..p, we care about 1..p-1
    root_of = [0] * (p + 1)
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x
            is_res[r] = 1
    
    # Compute prefix sum of is_res for O(1) range queries
    # prefix[i] = number of non-zero residues in [1, i]
    prefix = [0] * (p + 2)
    for i in range(1, p + 1):
        prefix[i] = prefix[i - 1] + is_res[i]
    
    # Scan s from 1 upward to find the minimal valid run of 5 consecutive residues
    # We need s, s+1, s+2, s+3, s+4 all to be non-zero residues
    # Note: s+4 <= 1008 (since s <= 1004), so no wraparound needed
    for s in range(1, 1005):
        if time.time() > deadline:
            # Should not happen for this small problem, but safety guard
            raise RuntimeError("Deadline exceeded during scan")
        
        # Check if all 5 values s..s+4 are residues
        # prefix[s+4] - prefix[s-1] should equal 5
        if s + 4 <= p - 1:
            if prefix[s + 4] - prefix[s - 1] == 5:
                roots = [root_of[s + i] for i in range(5)]
                return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero residues found')
# EVOLVE-BLOCK-END
