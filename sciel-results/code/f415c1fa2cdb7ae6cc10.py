"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    def is_valid_ap(c_new, c1, c2, c3):
        s = {c_new, c1, c2, c3}
        return len(s) >= 3

    def get_forbidden(idx, colors, K):
        forbidden = [False] * K
        max_d = idx // 3
        for d in range(1, max_d + 1):
            c1 = colors[idx - d]
            c2 = colors[idx - 2 * d]
            c3 = colors[idx - 3 * d]
            
            # Determine forbidden colors for this AP
            # We need len({c, c1, c2, c3}) >= 3
            # If c1==c2==c3: c != c1
            # If 2 distinct in c1,c2,c3: c not in {c1,c2,c3}
            # If 3 distinct in c1,c2,c3: no restriction
            
            distinct = {c1, c2, c3}
            if len(distinct) == 1:
                val = next(iter(distinct))
                if val < K:
                    forbidden[val] = True
            elif len(distinct) == 2:
                for val in distinct:
                    if val < K:
                        forbidden[val] = True
            # else len(distinct) == 3: no forbidden colors
            
        return forbidden

    def solve_with_K(K, max_nodes=500000):
        colors = [-1] * n
        nodes = 0
        
        def dfs(idx):
            nonlocal nodes
            if time.time() > deadline:
                return None
            
            if idx == n:
                return colors[:]
            
            nodes += 1
            if nodes > max_nodes:
                return None
            
            forbidden = get_forbidden(idx, colors, K)
            
            # Heuristic: Try colors that are not forbidden.
            # Order: 0 to K-1
            valid_colors = [c for c in range(K) if not forbidden[c]]
            
            # Randomize order to avoid getting stuck in one bad path repeatedly if we restart
            # But for a single run, deterministic is fine.
            # To help backtracking, we can try 'least constraining' but that's complex.
            # Let's just try in order.
            
            for c in valid_colors:
                colors[idx] = c
                res = dfs(idx + 1)
                if res is not None:
                    return res
                colors[idx] = -1
            
            return None
        
        return dfs(0)

    # Try K=3 first
    result = solve_with_K(3)
    if result is not None:
        return result
        
    # Try K=4
    result = solve_with_K(4)
    if result is not None:
        return result

    # Fallback: Greedy with randomization if DFS fails (should not happen for K=4)
    # Or just return a valid pattern if we know one.
    # A known valid pattern for 4 colors?
    # Let's try a simple greedy that picks random valid color to avoid dead ends.
    for attempt in range(100):
        colors = []
        success = True
        for idx in range(n):
            if time.time() > deadline:
                success = False
                break
            forbidden = get_forbidden(idx, colors, 128)
            valid = [c for c in range(128) if not forbidden[c]]
            if not valid:
                success = False
                break
            c = random.choice(valid)
            colors.append(c)
        if success and len(colors) == n:
            # Verify
            valid = True
            for i in range(n):
                for d in range(1, i//3 + 1):
                    if i - 3*d >= 0:
                        if len({colors[i], colors[i-d], colors[i-2*d], colors[i-3*d]}) < 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return colors

    # Last resort: return zeros (will score 0 but avoid crash)
    return [0] * n
# EVOLVE-BLOCK-END
