# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3*d
            if i3 < n:
                aps.append((a, a+d, a+2*d, i3))
    
    # For each position, list of (ap_idx, role)
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for role, pos in enumerate(ap):
            pos_to_aps[pos].append((idx, role))

    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3)
            if mask.bit_count() < 3:
                count += 1
        return count

    def compute_delta(i, old_color, new_color, colors):
        """Compute change in violation count when position i changes."""
        delta = 0
        for ap_idx, role in pos_to_aps[i]:
            ap = aps[ap_idx]
            # Get colors
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Old mask
            if role == 0:
                old_mask = (1 << old_color) | (1 << c1) | (1 << c2) | (1 << c3)
                new_mask = (1 << new_color) | (1 << c1) | (1 << c2) | (1 << c3)
            elif role == 1:
                old_mask = (1 << c0) | (1 << old_color) | (1 << c2) | (1 << c3)
                new_mask = (1 << c0) | (1 << new_color) | (1 << c2) | (1 << c3)
            elif role == 2:
                old_mask = (1 << c0) | (1 << c1) | (1 << old_color) | (1 << c3)
                new_mask = (1 << c0) | (1 << c1) | (1 << new_color) | (1 << c3)
            else:
                old_mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << old_color)
                new_mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << new_color)
                
            old_viol = old_mask.bit_count() < 3
            new_viol = new_mask.bit_count() < 3
            
            if old_viol and not new_viol:
                delta -= 1
            elif not old_viol and new_viol:
                delta += 1
        return delta

    best_result = None
    best_violations = float('inf')

    # Try small number of colors first
    for k in range(3, 8):
        if time.time() > deadline - 10:
            break
            
        # A few restarts
        max_restarts = 5 if k <= 4 else 3
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random init
            colors = [random.randint(0, k-1) for _ in range(n)]
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors
                
            best_colors_local = colors[:]
            best_violations_local = current_violations
            
            temp = 1.0
            cooling = 0.999
            steps = 0
            max_steps = 20000 # Reduced steps to fit time
            
            while steps < max_steps and time.time() < deadline - 2:
                steps += 1
                i = random.randint(0, n-1)
                old_color = colors[i]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                delta = compute_delta(i, old_color, new_color, colors)
                new_violations = current_violations + delta
                
                # Simulated Annealing acceptance
                if new_violations <= current_violations or (temp > 0.1 and random.random() < math.exp(-(new_violations - current_violations) / temp)):
                    colors[i] = new_color
                    current_violations = new_violations
                    if current_violations < best_violations_local:
                        best_violations_local = current_violations
                        best_colors_local = colors[:]
                    if current_violations == 0:
                        return colors
                    
                temp *= cooling
                
            if best_violations_local < best_violations:
                best_violations = best_violations_local
                best_result = best_colors_local

    # Fallback: Greedy construction
    if best_result is None or best_violations > 0:
        colors = [0] * n
        for x in range(n):
            found = False
            # Try colors 0 to 10
            for c in range(11):
                valid = True
                for d in range(1, x // 3 + 1):
                    i3 = x - 3*d
                    if i3 < 0:
                        continue
                    i2 = x - 2*d
                    i1 = x - d
                    c0, c1, c2 = colors[i3], colors[i1], colors[i2]
                    # Check if adding c creates < 3 distinct colors
                    mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c)
                    if mask.bit_count() < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    found = True
                    break
            if not found:
                colors[x] = 0 # Fallback, might be invalid but better than crash
            
        # Check if greedy worked
        v = count_violations(colors)
        if v < best_violations:
            best_result = colors
            best_violations = v

    if best_result is None:
        return [0] * n
        
    return best_result
# EVOLVE-BLOCK-END
