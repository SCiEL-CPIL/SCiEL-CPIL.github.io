import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 128
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(0, N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    def get_violations(colors):
        # Returns list of violating APs
        viol = []
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                viol.append(ap)
        return viol

    def count_violations(colors):
        return len(get_violations(colors))

    def optimize(k, time_limit):
        # Initialize
        colors = [random.randint(0, k - 1) for _ in range(N)]
        cur_viol = get_violations(colors)
        
        if not cur_viol:
            return colors

        best_colors = colors[:]
        best_viol_count = len(cur_viol)

        while time.time() < time_limit:
            if not cur_viol:
                return colors

            # Pick a random violating AP
            ap = random.choice(cur_viol)
            
            # Try to fix it by changing one of its 4 positions
            # Pick a random position in the AP
            pos_idx = random.choice(ap)
            old_color = colors[pos_idx]
            
            best_new_color = old_color
            best_new_viol_count = len(cur_viol)
            
            # Try all k colors
            for c in range(k):
                if c == old_color:
                    continue
                colors[pos_idx] = c
                # Recalculate violations? 
                # To be efficient, we could update only affected APs.
                # But N=128, APs~2700. Recalculation is ~0.1ms.
                # k is small (3-5). So 5 recalcs per step. ~0.5ms per step.
                # 2000 steps/sec. Plenty of time.
                new_viol = get_violations(colors)
                new_count = len(new_viol)
                
                if new_count < best_new_viol_count:
                    best_new_viol_count = new_count
                    best_new_color = c
                    best_new_viol_list = new_viol
                    # Early exit if we found a fix that resolves this AP and doesn't break others?
                    # No, just track min.
                
            # Apply best color
            colors[pos_idx] = best_new_color
            cur_viol = get_violations(colors) # Re-fetch to be safe, or use best_new_viol_list
            
            if len(cur_viol) < best_viol_count:
                best_viol_count = len(cur_viol)
                best_colors = colors[:]
                if best_viol_count == 0:
                    return best_colors
            
            # If stuck (no improvement in a while), random restart or random jump
            # Simple heuristic: if best_new_viol_count == best_viol_count (no improvement),
            # maybe accept a random move to escape?
            # For now, let's just keep trying. The random choice of AP and pos helps.
            
            # Additional: If we haven't improved in many steps, restart
            # We don't track steps here easily without a counter.
            # Let's add a step counter.
            
        return best_colors

    # Add step counter to optimize for restarts
    def optimize_with_restarts(k, time_limit):
        colors = [random.randint(0, k - 1) for _ in range(N)]
        cur_viol = get_violations(colors)
        
        if not cur_viol:
            return colors

        best_colors = colors[:]
        best_viol_count = len(cur_viol)
        steps_since_improve = 0
        max_steps_no_improve = 2000

        while time.time() < time_limit:
            if not cur_viol:
                return colors

            ap = random.choice(cur_viol)
            pos_idx = random.choice(ap)
            old_color = colors[pos_idx]
            
            best_new_color = old_color
            best_new_viol_count = len(cur_viol)
            best_new_viol_list = cur_viol
            
            for c in range(k):
                if c == old_color:
                    continue
                colors[pos_idx] = c
                new_viol = get_violations(colors)
                new_count = len(new_viol)
                
                if new_count < best_new_viol_count:
                    best_new_viol_count = new_count
                    best_new_color = c
                    best_new_viol_list = new_viol
            
            colors[pos_idx] = best_new_color
            cur_viol = best_new_viol_list
            
            if len(cur_viol) < best_viol_count:
                best_viol_count = len(cur_viol)
                best_colors = colors[:]
                steps_since_improve = 0
                if best_viol_count == 0:
                    return best_colors
            else:
                steps_since_improve += 1
                
            if steps_since_improve > max_steps_no_improve:
                # Restart
                colors = [random.randint(0, k - 1) for _ in range(N)]
                cur_viol = get_violations(colors)
                if len(cur_viol) < best_viol_count:
                    best_viol_count = len(cur_viol)
                    best_colors = colors[:]
                    if best_viol_count == 0:
                        return best_colors
                steps_since_improve = 0

        return best_colors

    for k in range(3, 8):
        time_limit = min(deadline, time.time() + 10.0)
        result = optimize_with_restarts(k, time_limit)
        
        if result is not None and count_violations(result) == 0:
            return result
            
    # Fallback
    colors = [i % 100 for i in range(N)]
    if count_violations(colors) != 0:
        colors = list(range(N))
    return colors
# EVOLVE-BLOCK-END
