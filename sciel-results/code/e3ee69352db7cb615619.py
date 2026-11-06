"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _all_4tps(n=128):
    """Generate all non-constant 4-term arithmetic progressions in 0..n-1."""
    tps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                tps.append((a, b, c, e))
    return tps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    tps = _all_4tps(n)
    num_tps = len(tps)
    
    # Pre-compute for each position, which 4-TPs it participates in
    pos_tps = [[] for _ in range(n)]
    for i, tp in enumerate(tps):
        for pos in tp:
            pos_tps[pos].append(i)
    
    def count_violations(colors):
        """Count the number of 4-TPs with fewer than 3 distinct colors."""
        viol_count = 0
        for tp in tps:
            c0, c1, c2, c3 = colors[tp[0]], colors[tp[1]], colors[tp[2]], colors[tp[3]]
            if len({c0, c1, c2, c3}) < 3:
                viol_count += 1
        return viol_count

    def local_search(k_colors, time_limit):
        """
        Perform local search to find a valid coloring with k_colors.
        Returns (colors, is_valid).
        """
        start_time_ls = time.time()
        best_colors = None
        best_viol = float('inf')
        
        # Multiple restarts
        while time.time() < start_time_ls + time_limit:
            # Initialize with random assignment
            colors = [random.randint(0, k_colors - 1) for _ in range(n)]
            
            # Initialize violation state
            is_violated = [False] * num_tps
            viol_count = 0
            for i, tp in enumerate(tps):
                if len({colors[tp[0]], colors[tp[1]], colors[tp[2]], colors[tp[3]]}) < 3:
                    is_violated[i] = True
                    viol_count += 1
            
            if viol_count < best_viol:
                best_viol = viol_count
                best_colors = list(colors)
                if viol_count == 0:
                    return best_colors, True
            
            # Hill Climbing
            improved = True
            step_count = 0
            max_steps = 10000
            while improved and step_count < max_steps:
                if time.time() > start_time_ls + time_limit:
                    break
                
                improved = False
                step_count += 1
                
                # Find a violated TP
                # To be efficient, we can pick a random position and check its TPs,
                # but picking a random violated TP directly is better if we maintain a list.
                # For simplicity and speed with small N, we scan for one.
                # Optimization: Maintain a list of violated indices?
                # Let's just pick a random index and check. If M is ~2000, this is fast.
                # But if viol_count is high, we want to pick a violated one.
                
                # Efficient approach: Pick a random position, then a random TP containing it.
                # If that TP is violated, we work on it.
                # This might miss some violations but is O(1) per step.
                
                # Let's stick to finding a specific violated TP for better convergence.
                # Since num_tps is small, linear scan is acceptable.
                violated_idx = -1
                for i in range(num_tps):
                    if is_violated[i]:
                        violated_idx = i
                        break
                
                if violated_idx == -1:
                    # No violations
                    return colors, True
                
                tp = tps[violated_idx]
                pos = random.choice(tp)
                old_color = colors[pos]
                new_color = random.randint(0, k_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Update violation counts for TPs containing pos
                delta_viol = 0
                affected_tps = pos_tps[pos]
                
                # Store old states to revert if needed
                old_states = {}
                for i in affected_tps:
                    old_states[i] = is_violated[i]
                    
                for i in affected_tps:
                    t = tps[i]
                    is_now_viol = len({colors[t[0]], colors[t[1]], colors[t[2]], colors[t[3]]}) < 3
                    if old_states[i] and not is_now_viol:
                        delta_viol -= 1
                    elif not old_states[i] and is_now_viol:
                        delta_viol += 1
                    is_violated[i] = is_now_viol
                        
                new_viol = viol_count + delta_viol
                
                if new_viol < viol_count:
                    viol_count = new_viol
                    improved = True
                    if viol_count < best_viol:
                        best_viol = viol_count
                        best_colors = list(colors)
                        if viol_count == 0:
                            return best_colors, True
                else:
                    # Revert
                    colors[pos] = old_color
                    for i in affected_tps:
                        is_violated[i] = old_states[i]
                        
            # End while improved
            
        return best_colors, (best_viol == 0)

    # Try 3 colors first
    if time.time() < deadline - 30.0:
        res, valid = local_search(3, 25.0)
        if valid:
            return res
            
    # Try 4 colors
    if time.time() < deadline - 10.0:
        res, valid = local_search(4, 25.0)
        if valid:
            return res

    # Fallback: Try 5 colors
    if time.time() < deadline - 5.0:
        res, valid = local_search(5, 5.0)
        if valid:
            return res

    # Ultimate Fallback: Greedy construction
    # This ensures validity but might use more colors.
    # A simple valid coloring: color[i] = i % 4 ?
    # Let's check if 4 colors always work with a specific pattern?
    # No, just return a high-color valid solution.
    # Using 128 colors is trivially valid.
    return list(range(128))

# EVOLVE-BLOCK-END
