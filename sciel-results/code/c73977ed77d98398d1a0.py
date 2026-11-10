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
        cands = [[] for _ in range(n)]
        
        for i in range(1, n + 1):
            first_mult = (m // i + 1) * i
            end_val = m + L
            for v in range(first_mult, end_val + 1, i):
                cands[i - 1].append(v)
        
        # Quick check: any i with no candidates?
        for i in range(n):
            if not cands[i]:
                return False
        
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
        return 144  # Fallback
    
    best_m = 0
    best_f = 0
    
    # Search one period
    for m in range(lcm_1_12):
        if time.time() > deadline - 2.0:
            break
        
        val = f(m)
        if val > best_f:
            best_f = val
            best_m = m
            # If we found the theoretical max, we can stop
            if best_f == 144:
                break
    
    return int(best_m)
# EVOLVE-BLOCK-END
