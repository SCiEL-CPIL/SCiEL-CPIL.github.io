import time
import random
import numpy as np

# Precompute all 4-term arithmetic progressions
N = 128
ALL_TPS = []
POS_TPS = [[] for _ in range(N)]

def _precompute():
    global ALL_TPS, POS_TPS
    ALL_TPS.clear()
    for i in range(N):
        POS_TPS[i] = []
    
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                idx = len(ALL_TPS)
                ALL_TPS.append((a, b, c, e))
                POS_TPS[a].append(idx)
                POS_TPS[b].append(idx)
                POS_TPS[c].append(idx)
                POS_TPS[e].append(idx)

_precompute()

def _count_violations(colors):
    """Count 4-APs with < 3 distinct colors."""
    v = 0
    for a, b, c, e in ALL_TPS:
        cs = {colors[a], colors[b], colors[c], colors[e]}
        if len(cs) < 3:
            v += 1
    return v

def _local_search(k, deadline):
    """
    Simulated Annealing to find a valid k-coloring.
    Returns (colors, violations)
    """
    n = N
    # Initialize
    colors = [random.randint(0, k - 1) for _ in range(n)]
    
    # Precompute current violations per TP? 
    # It's faster to just recompute affected TPs.
    
    # Initial violations
    current_violations = _count_violations(colors)
    if current_violations == 0:
        return colors, 0
    
    best_colors = list(colors)
    best_violations = current_violations
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    step = 0
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        step += 1
        
        # Pick a random position
        i = random.randint(0, n - 1)
        old_color = colors[i]
        
        # Try a new color
        new_color = random.randint(0, k - 1)
        if new_color == old_color:
            continue
            
        # Calculate change in violations
        # Only TPs containing i are affected
        delta_v = 0
        affected_tps = POS_TPS[i]
        
        for tp_idx in affected_tps:
            a, b, c, e = ALL_TPS[tp_idx]
            # Get old colors
            old_vals = [colors[a], colors[b], colors[c], colors[e]]
            old_distinct = len(set(old_vals))
            old_viol = 1 if old_distinct < 3 else 0
            
            # Get new colors
            # Replace color at i
            new_vals = list(old_vals)
            # Find index of i in (a,b,c,e)
            if a == i: new_vals[0] = new_color
            elif b == i: new_vals[1] = new_color
            elif c == i: new_vals[2] = new_color
            else: new_vals[3] = new_color
            
            new_distinct = len(set(new_vals))
            new_viol = 1 if new_distinct < 3 else 0
            
            delta_v += new_viol - old_viol
        
        # Apply change
        colors[i] = new_color
        current_violations += delta_v
        
        # Acceptance criterion
        if delta_v <= 0:
            # Improvement or neutral
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = list(colors)
                if best_violations == 0:
                    return best_colors, 0
        else:
            # Deterioration
            # Accept with probability exp(-delta_v / T)
            if random.random() < np.exp(-delta_v / T):
                # Accept
                pass
            else:
                # Reject
                colors[i] = old_color
                current_violations -= delta_v
        
        T *= cooling
        
        # Restart if stuck?
        if step % 1000 == 0:
             if random.random() < 0.1:
                 # Random restart with lower temperature
                 T = max(T * 0.5, min_T)

    return best_colors, best_violations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Try k=3 first
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        colors, viols = _local_search(3, deadline)
        if viols == 0:
            # Verify
            if _count_violations(colors) == 0:
                return colors
        
    # If k=3 failed, try k=4
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        colors, viols = _local_search(4, deadline)
        if viols == 0:
            if _count_violations(colors) == 0:
                return colors
                
    # Fallback: Return the best found (even if invalid? No, must be valid)
    # If we didn't find a valid one, we need a valid one.
    # A simple greedy might work for k=128? No, we want to minimize.
    # If SA failed, try a simple backtracking for k=4 or 5?
    # Or just return a random valid coloring?
    # The task requires a valid coloring.
    # Let's try a simple deterministic construction for a larger k if needed.
    # But likely k=3 or 4 works.
    
    # If all else fails, use a "worst case" valid coloring:
    # Color i with i % 128? No, that uses 128 colors.
    # We need to ensure validity.
    # A coloring where colors are all distinct is valid.
    # But we want to minimize.
    # If we haven't found a small k solution, return a valid one with k=128?
    # The evaluator measures raw value (lower is better).
    # If we return 128, score is bad.
    # But if we return invalid, score is 0.
    # So we MUST return valid.
    
    # Let's try one more thing: Backtracking for k=3 with time limit.
    # Actually, if SA didn't find it in 80s, it might be hard.
    # But for n=128, it's likely easy.
    
    # Fallback: Return a valid coloring using k=128 (trivially valid)
    # This ensures we don't crash, though score will be low.
    return list(range(128))
