"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    N = 12
    MAX_M = 1000000
    DEADLINE = 85.0  # safety margin before 90s limit
    
    start_time = time.time()
    
    PERIOD = 27720
    
    # Precompute adjacency bitmasks for each residue class
    # For a given m, the window is (m, m+L], values m+1 to m+L
    # Value v = m + 1 + j (j from 0 to L-1) is divisible by i iff (m+1+j) % i == 0
    # So j % i == (i - (m+1) % i) % i
    # Precompute for each residue r = m % PERIOD, and for each i, the set of j in [0, 143] where (r+1+j) % i == 0
    
    # Actually, let's precompute for each r in [0, PERIOD), for each i in 1..12,
    # a bitmask of positions j in 0..143 where (r + 1 + j) % i == 0
    # But 27720 * 12 * 144 bits = 27720 * 12 * 18 bytes = ~6MB, feasible
    
    # Better: for each r, for each i, compute the bitmask once
    # bitmask[i] has bit j set iff (r + 1 + j) % i == 0 for j in 0..143
    
    # Precompute all masks
    masks = {}  # masks[r] = list of 12 integers, each a 144-bit mask
    for r in range(PERIOD):
        row = []
        for i in range(1, N + 1):
            mask = 0
            # j from 0 to 143: check if (r + 1 + j) % i == 0
            # (r+1+j) % i == 0 iff j % i == (-1 - r) % i
            target = ((-1 - r) % i)
            # j values: target, target+i, target+2i, ... up to 143
            for j in range(target, 144, i):
                mask |= (1 << j)
            row.append(mask)
        masks[r] = row
    
    def check_matching_fast(r, L):
        """Check if perfect matching exists for window (m, m+L] where m % PERIOD = r."""
        # L <= 144, use bitmasks
        # adj[i] = bitmask of valid positions for left node i
        # We only care about positions 0..L-1
        
        # Build adjacency as lists of positions for matching
        adj = [[] for _ in range(N)]
        for i in range(N):
            mask = masks[r][i] & ((1 << L) - 1)  # only positions 0..L-1
            # Extract set bits
            m = mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                pos = lb.bit_length() - 1
                adj[i].append(pos)
                m ^= lb
        
        # BFS-based bipartite matching (Hopcroft-Karp simplified for small N)
        # Since N=12 is tiny, use simple DFS with iterative approach
        match_r = [-1] * L  # match_r[j] = left node
        match_l = [-1] * N  # match_l[i] = right node
        
        # Use iterative DFS to avoid recursion overhead
        def try_match(u):
            visited = [False] * L
            stack = [(u, 0)]  # (node, next_adj_index)
            path = []
            while stack:
                node, idx = stack[-1]
                if idx < len(adj[node]):
                    j = adj[node][idx]
                    stack[-1] = (node, idx + 1)
                    if visited[j]:
                        continue
                    visited[j] = True
                    if match_r[j] == -1:
                        # Found augmenting path
                        # Reconstruct path
                        # The path is in the stack
                        # Actually, let's use a different approach
                        pass
                    else:
                        stack.append((match_r[j], 0))
                        path.append((node, j))
                else:
                    stack.pop()
                    if path and path[-1][0] == node:
                        path.pop()
            # This is getting complicated. Let's use a cleaner iterative DFS.
            return False
        
        # Simpler: use recursive DFS but with manual stack
        # For N=12, recursion depth is at most 12, which is fine
        
        def dfs(u, visited):
            for j in adj[u]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_r[j] == -1 or dfs(match_r[j], visited):
                    match_r[j] = u
                    match_l[u] = j
                    return True
            return False
        
        count = 0
        for u in range(N):
            visited = [False] * L
            if dfs(u, visited):
                count += 1
            else:
                return False  # Early exit
        
        return True
    
    def compute_f(m):
        """Compute f(12, m) = least positive L such that matching exists."""
        r = m % PERIOD
        # Linear scan from L=1 upward, early exit
        for L in range(1, 145):
            if check_matching_fast(r, L):
                return L
        return -1
    
    best_m = 0
    best_L = 0
    last_check = start_time
    
    for m in range(PERIOD):
        # Time check every 1000 iterations
        if (m & 0x3FF) == 0:
            now = time.time()
            if now - start_time > DEADLINE - 5:
                break
        
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
