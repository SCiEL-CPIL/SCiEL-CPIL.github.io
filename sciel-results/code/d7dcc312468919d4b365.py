"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    lcm_1_12 = 27720  # lcm(1..12)
    
    def can_assign(m, L):
        """Check if we can assign distinct b_i in (m, m+L] with i | b_i."""
        # Build candidates for each i in 1..12
        # cands[i-1] is list of values in (m, m+L] divisible by i
        cands = [[] for _ in range(n)]
        # To avoid modulo in inner loop, we can calculate start multiple
        # for i in 1..n:
        #   first_mult = (m // i + 1) * i
        #   for v in range(first_mult, m + L + 1, i):
        #       cands[i-1].append(v)
        
        for i in range(1, n + 1):
            first_mult = (m // i + 1) * i
            # Range is (m, m+L]
            # So v > m and v <= m+L
            end_val = m + L
            for v in range(first_mult, end_val + 1, i):
                cands[i - 1].append(v)
        
        # Kuhn's algorithm for bipartite matching
        # Left: 0..11 (indices for i=1..12)
        # Right: values in (m, m+L]
        # match_right: value -> left_index
        match_right = {}
        
        def dfs(left, visited):
            i = left + 1
            for v in cands[left]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_right or dfs(match_right[v], visited):
                    match_right[v] = left
                    return True
            return False

        count = 0
        for left in range(n):
            visited = set()
            if dfs(left, visited):
                count += 1
            else:
                return False
        return True

    def f(m):
        """Compute f(12, m) = least positive L such that assignment exists."""
        # L is at most 144 per task description, but let's be safe up to 200
        # We iterate L from 1 upwards. Since we want to MAXIMIZE f(m), 
        # and f(m) is the MIN L, a high f(m) means we fail for many small L.
        # So for the "best" m, this loop runs long.
        # For "bad" m, it runs short.
        for L in range(1, 201):
            if can_assign(m, L):
                return L
        return 200  # Fallback

    best_m = 0
    best_f = 0
    
    # Since f(m) is periodic with period 27720, we only need to search one period
    # 0 <= m < 27720
    # 27720 iterations.
    
    # Optimization: Check time periodically
    for m in range(0, lcm_1_12):
        if time.time() > deadline - 2.0:
            break
        
        val = f(m)
        if val > best_f:
            best_f = val
            best_m = m

    return int(best_m)
# EVOLVE-BLOCK-END
