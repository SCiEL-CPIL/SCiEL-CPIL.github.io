"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    LCM = 27720  # lcm(1..12)
    
    def can_assign(m, L):
        """Check if there's a perfect matching: assign distinct b_i in (m, m+L] with i | b_i."""
        # Build adjacency as bitmask for each index i (1..12)
        # Position pos in [0, L-1] corresponds to value m+1+pos
        adj = [0] * n  # bitmask of valid positions for each index
        for i in range(n):
            idx = i + 1
            # First multiple of idx in (m, m+L]: smallest v > m with idx|v
            first = ((m + 1 + idx - 1) // idx) * idx
            if first > m + L:
                continue
            mask = 0
            for v in range(first, m + L + 1, idx):
                pos = v - (m + 1)
                if 0 <= pos < L:
                    mask |= (1 << pos)
            adj[i] = mask
        
        # Check individual indices have at least one option
        for i in range(n):
            if adj[i] == 0:
                return False
        
        # Use bitmask-based bipartite matching with backtracking
        # match[i] = position assigned to index i, or -1
        match = [-1] * n  # match[i] = pos for index i
        # pos_owner[pos] = index that owns pos, or -1
        pos_owner = [-1] * L
        
        def try_assign(i, visited):
            """Try to assign a position to index i using backtracking."""
            mask = adj[i]
            # Try positions in order
            while mask:
                pos = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1  # clear lowest set bit
                if visited & (1 << pos):
                    continue
                visited |= (1 << pos)
                if pos_owner[pos] == -1:
                    pos_owner[pos] = i
                    match[i] = pos
                    return True
                else:
                    # Try to reassign the current owner
                    old_owner = pos_owner[pos]
                    if try_assign(old_owner, visited):
                        pos_owner[pos] = i
                        match[i] = pos
                        return True
            return False
        
        for i in range(n):
            if not try_assign(i, 0):
                return False
        return True
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        # f(12,m) depends only on m mod LCM
        m_mod = m % LCM
        lo, hi = 12, 144
        if not can_assign(m_mod, hi):
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m_mod, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Key insight: f(12,m) depends only on m mod 27720
    # So we only need to check 27720 distinct residue classes
    # But we can be smarter: many residues will have similar f values
    # Let's exhaustively search all residues, but with early termination
    
    best_f = -1
    best_m = 0
    
    # First pass: check all residues 0..27719
    # But we can optimize by checking in order of likelihood
    # Residues near 0 mod small numbers tend to have larger f
    
    # Process residues in batches, checking time periodically
    for m_mod in range(LCM):
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(m_mod)
        if f_val > best_f:
            best_f = f_val
            best_m = m_mod  # m_mod is already in [0, LCM-1] subset of [0, max_m]
    
    # If we found a good value, return the smallest m in [0, max_m] with that residue
    # m_mod is already valid since 0 <= m_mod < 27720 <= 1000000
    return int(best_m)
# EVOLVE-BLOCK-END
