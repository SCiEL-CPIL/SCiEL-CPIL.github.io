import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute all 4-term APs: (i, i+d, i+2d, i+3d) with 0 <= i < N
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps4)
    
    # Map each position to the APs it participates in
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps4):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def initial_violations(colors):
        """Calculate number of violating APs."""
        v = 0
        for ap in aps4:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                v += 1
        return v
    
    def try_local_search(num_colors, max_restarts=20, max_iters_per_restart=50000):
        """
        Attempt to find a valid coloring using exactly num_colors colors (0 to num_colors-1).
        Uses Simulated Annealing / Hill Climbing.
        """
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Random initialization
            colors = [random.randint(0, num_colors - 1) for _ in range(N)]
            
            # Calculate initial energy
            energy = initial_violations(colors)
            
            if energy == 0:
                return colors
            
            # Local Search Loop
            current_energy = energy
            best_energy = energy
            best_colors = colors[:]
            
            # Temperature for simulated annealing
            T = 1.0
            cooling_rate = 0.9995
            
            for it in range(max_iters_per_restart):
                if time.time() - start_time > DEADLINE:
                    return best_colors if best_energy == 0 else None
                
                if current_energy == 0:
                    return colors
                
                # Pick a random position
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                
                # Pick a new color different from old
                if num_colors > 1:
                    new_color = random.randint(0, num_colors - 1)
                    if new_color == old_color:
                        new_color = (new_color + 1) % num_colors
                else:
                    break
                
                # Calculate energy change
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps4[ap_idx]
                    # Get colors of the AP before change
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    distinct_before = len({c0, c1, c2, c3})
                    was_violating = distinct_before < 3
                    
                    # Calculate distinct_after
                    c_set = set()
                    for p in ap:
                        if p == pos:
                            c_set.add(new_color)
                        else:
                            c_set.add(colors[p])
                    distinct_after = len(c_set)
                    is_violating = distinct_after < 3
                    
                    if was_violating and not is_violating:
                        delta -= 1
                    elif not was_violating and is_violating:
                        delta += 1
                
                new_energy = current_energy + delta
                
                # Acceptance criterion
                if new_energy <= current_energy:
                    colors[pos] = new_color
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors[:]
                else:
                    # Accept worse move with probability exp(-delta/T)
                    if random.random() < (2.71828 ** (-delta / T)):
                        colors[pos] = new_color
                        current_energy = new_energy
                
                # Cool down
                T *= cooling_rate
                if T < 0.01:
                    T = 0.01
            
            if best_energy == 0:
                return best_colors
                
        return None

    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Increase restarts for smaller color counts as they are harder
        restarts = 50 if num_colors == 3 else 20
        iters = 20000 if num_colors == 3 else 50000
        
        result = try_local_search(num_colors, max_restarts=restarts, max_iters_per_restart=iters)
        
        if result is not None:
            # Verify
            valid = True
            for ap in aps4:
                if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                    valid = False
                    break
            if valid:
                return result
        
    # Fallback: Greedy construction minimizing colors used dynamically
    colors = []
    for x in range(1, N + 1):
        used_colors = set(colors)
        
        candidates = list(used_colors)
        random.shuffle(candidates)
        
        assigned = False
        for c in candidates:
            if is_greedy_valid(colors, x, c):
                colors.append(c)
                assigned = True
                break
        
        if not assigned:
            for c in range(128):
                if c in used_colors:
                    continue
                if is_greedy_valid(colors, x, c):
                    colors.append(c)
                    assigned = True
                    break
        
        if not assigned:
            colors.append(0)
            
    return colors

def is_greedy_valid(colors, x, c):
    """
    Check if assigning color c to position x (1-indexed) is valid.
    colors contains colors for positions 1 to x-1.
    """
    for d in range(1, (x - 1) // 3 + 1):
        if x - 3 * d < 1:
            break
            
        c1 = colors[x - 3 * d - 1]
        c2 = colors[x - 2 * d - 1]
        c3 = colors[x - d - 1]
        
        distinct = len({c, c1, c2, c3})
        if distinct < 3:
            return False
            
    return True
# EVOLVE-BLOCK-END
