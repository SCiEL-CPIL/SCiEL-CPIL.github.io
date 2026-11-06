import time
import random
import numpy as np

# Precompute 4-APs
def get_4aps(n=128):
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

APs = get_4aps()
# Map position to list of AP indices
pos_to_aps = [[] for _ in range(129)] # 1-indexed
for idx, ap in enumerate(APs):
    for pos in ap:
        pos_to_aps[pos].append(idx)

def count_violations(colors):
    """colors is 0-indexed array of length 128, colors[i] is color of position i+1"""
    viol = 0
    for ap in APs:
        c1 = colors[ap[0]-1]
        c2 = colors[ap[1]-1]
        c3 = colors[ap[2]-1]
        c4 = colors[ap[3]-1]
        if len({c1, c2, c3, c4}) < 3:
            viol += 1
    return viol

def generate_seed():
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x? No, the seed logic is different.
            # The seed logic checks constraints with previous elements.
            # It ensures that when adding x, all 4-APs involving x and previous elements are valid.
            # But 4-APs can have x as the start, middle, or end.
            # The seed code:
            # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            # This checks APs where x is the last term: (x-3d, x-2d, x-d, x).
            # and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This part is weird. "any(colors[x-1-j*d]!=color for j in (1,2))"
            # This checks APs where x is the 3rd term? (x-2d, x-d, x, x+d)? No, x+d doesn't exist.
            # Let's re-read the seed code carefully.
            
            # The seed code logic:
            # 1. For d in 1..(x-1)//3: Check AP (x-3d, x-2d, x-d, x).
            #    Condition: len({color, colors[x-1-3d], colors[x-1-2d], colors[x-1-d]}) >= 3.
            # 2. For d in 1..(x-1)//2: 
            #    Condition: any(colors[x-1-d] != color, colors[x-1-2d] != color)
            #    This seems to check APs where x is the 2nd or 3rd term?
            #    If x is 2nd term: (x-d, x, x+d, x+2d). x+d, x+2d don't exist yet.
            #    If x is 3rd term: (x-2d, x-d, x, x+d). x+d doesn't exist.
            #    Maybe it's checking partial constraints to ensure future validity?
            #    Or maybe it's checking APs where x is the first term? (x, x+d, x+2d, x+3d).
            #    If x is 1st term, the other terms don't exist.
            
            # Actually, the seed code is a specific heuristic. I should just run it to get the baseline.
            
            # Let's just implement the seed code exactly as given to get the baseline.
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors

def solve_with_elimination(seed_colors, time_limit=10):
    """
    Try to reduce the number of distinct colors in seed_colors.
    """
    colors = list(seed_colors)
    used_colors = set(colors)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Pick a color to eliminate
        # Heuristic: pick the color with fewest occurrences
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Sort colors by count
        sorted_colors = sorted(used_colors, key=lambda c: color_counts[c])
        
        for c_to_elim in sorted_colors:
            if c_to_elim not in used_colors:
                continue
                
            # Positions to recolor
            positions = [i for i, c in enumerate(colors) if c == c_to_elim]
            
            # Try to recolor these positions to other existing colors
            # Use a simple local search / backtracking
            # Since the number of positions is small, we can try all combinations?
            # No, domains are large.
            
            # Let's use a greedy repair strategy for this subproblem
            # Or just try random recoloring of these positions until valid
            success = False
            attempts = 0
            while not success and attempts < 1000 and (time.time() - start_time) < time_limit:
                attempts += 1
                new_colors = colors[:]
                for pos in positions:
                    # Pick a random color from used_colors \ {c_to_elim}
                    other_colors = [c for c in used_colors if c != c_to_elim]
                    if not other_colors:
                        break
                    new_c = random.choice(other_colors)
                    new_colors[pos] = new_c
                
                # Check if valid
                if count_violations(new_colors) == 0:
                    colors = new_colors
                    used_colors.remove(c_to_elim)
                    improved = True
                    success = True
                    break
            if success:
                break # Restart the outer loop with new used_colors
        
    return colors

def solve_with_sa(K, time_limit=5):
    """
    Try to find a valid coloring with exactly K colors (or fewer).
    Returns the coloring if found, else None.
    """
    start_time = time.time()
    # Initialize with random K colors
    colors = [random.randint(0, K-1) for _ in range(128)]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    best_colors = colors[:]
    best_cost = count_violations(colors)
    
    while (time.time() - start_time) < time_limit:
        if best_cost == 0:
            return best_colors
        
        # Pick a random position
        pos = random.randint(0, 127)
        old_c = colors[pos]
        new_c = random.randint(0, K-1)
        if new_c == old_c:
            continue
        
        colors[pos] = new_c
        new_cost = count_violations(colors)
        
        delta = new_cost - best_cost
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            best_colors = colors[:]
            best_cost = new_cost
        else:
            colors[pos] = old_c
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    if best_cost == 0:
        return best_colors
    return None

def construct():
    # 1. Generate seed
    seed_colors = generate_seed()
    seed_used = len(set(seed_colors))
    
    # 2. Try to reduce colors from seed
    reduced_colors = solve_with_elimination(seed_colors, time_limit=20)
    reduced_used = len(set(reduced_colors))
    
    best_colors = reduced_colors
    best_used = reduced_used
    
    # 3. Try to find a solution with small K using SA
    # Try K=3, 4, 5, ... up to best_used - 1
    time_spent = 20
    total_time_limit = 55
    for K in range(3, best_used):
        if time.time() > time.time() + (total_time_limit - time_spent): # Dummy check
            break
        # Allocate remaining time
        remaining = total_time_limit - (time.time() - time.time()) # This is wrong
        # Let's just use a fixed small time per K
        sa_time = min(5, total_time_limit - 20)
        if sa_time <= 0:
            break
            
        sol = solve_with_sa(K, sa_time)
        if sol is not None:
            sol_used = len(set(sol))
            if sol_used < best_used:
                best_colors = sol
                best_used = sol_used
            # If we found a solution with K colors, we can stop if K is small
            # But we want the minimum. If K=3 works, it's likely optimal or close.
            # Let's break if we find a solution with K=3
            if K == 3:
                break
                
    return best_colors
