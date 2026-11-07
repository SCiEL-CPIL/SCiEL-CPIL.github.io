"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

PERIOD = 27720  # LCM(1..12)

def has_valid_set_fast(m, L):
    """Fast check if (m, m+L] contains distinct b_1,...,b_12 with i|b_i."""
    # For each i in 1..12, find which positions j in [0,L) satisfy (m+1+j) % i == 0
    # Use bitsets for speed
    # Position j: number is m+1+j, valid for index i if (m+1+j) % i == 0
    
    # Build adjacency as bitsets
    adj = [0] * 12
    for i in range(12):
        idx = i + 1
        r = (idx - (m + 1) % idx) % idx
        # Set bits at positions r, r+idx, r+2*idx, ... < L
        mask = 0
        for j in range(r, L, idx):
            mask |= (1 << j)
        adj[i] = mask
    
    # Kuhn's algorithm with bit operations
    match_right = [-1] * L  # which left node is matched to right node j
    
    def dfs(u, seen):
        # Get available right nodes for u not yet seen
        avail = adj[u] & ~seen
        while avail:
            # Get lowest set bit
            v = (avail & (-avail)).bit_length() - 1
            avail &= avail - 1  # clear lowest set bit
            seen |= (1 << v)
            if match_right[v] == -1 or dfs(match_right[v], seen):
                match_right[v] = u
                return True
        return False
    
    for u in range(12):
        if not dfs(u, 0):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # For each L from 12 to 144, find which residues r in [0, PERIOD) have valid matchings
    # f(12, r) = smallest L such that r is "good" at L
    # We want to maximize f(12, r)
    
    # Track for each residue, the minimal L where it becomes good
    min_L_for_residue = [145] * PERIOD  # 145 means not found (impossible, but safe)
    
    for L in range(12, 145):
        if time.time() > deadline:
            break
        # For each residue r, check if has_valid_set_fast(r, L) is True
        # If yes and min_L_for_residue[r] == 145, set it to L
        for r in range(PERIOD):
            if min_L_for_residue[r] == 145:
                if has_valid_set_fast(r, L):
                    min_L_for_residue[r] = L
    
    # Find the residue with the maximum min_L
    best_r = 0
    best_val = 0
    for r in range(PERIOD):
        if min_L_for_residue[r] > best_val:
            best_val = min_L_for_residue[r]
            best_r = r
    
    m_result = best_r
    if m_result > 1000000:
        m_result = 0
    
    return int(m_result)

# EVOLVE-BLOCK-END
