"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    lcm_1_12 = 27720  # lcm(1..12)
    
    def can_assign(m, L):
        """Check if we can assign distinct b_i in (m, m+L] with i | b_i."""
        # Build candidates for each i in 1..12
        cands = [[] for _ in range(n)]
        
        for i in range(1, n + 1):
            first_mult = (m // i + 1) * i
            end_val = m + L
            for v in range(first_mult, end_val + 1, i):
                cands[i - 1].append(v)
        
        # Kuhn's algorithm for bipartite matching
        match_right = {}
        
        def dfs(left, visited):
            for v in cands[left]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_right or dfs(match_right[v], visited):
                    match_right[v] = left
                    return True
            return False

        for left in range(n):
            visited = set()
            if not dfs(left, visited):
                return False
        return True
    
    def f(m):
        """Compute f(12, m) = least positive L such that assignment exists."""
        for L in range(1, 145):
            if can_assign(m, L):
                return L
        return 144  # Fallback (max per task description)
    
    # Since f(m) is periodic with period 27720, search one period
    # Use time guard to avoid timeout
    best_m = 12
    best_f = f(12)
    
    for m in range(0, lcm_1_12):
        if time.time() > deadline - 2.0:
            break
        
        val = f(m)
        if val > best_f:
            best_f = val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
