import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 140  # 150s wall-clock limit, keep 10s margin
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed
    # a + 3d <= n-1 => a <= n-1-3d, d >= 1
    ap_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each position, the list of (ap_index, other_positions)
    # For fast delta computation
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for i, pos in enumerate(ap):
            others = [ap[j] for j in range(4) if j != i]
            pos_aps[pos].append((idx, tuple(others)))
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if coloring[a] == coloring[b] and coloring[c] == coloring[d] and coloring[a] == coloring[c]:
                count += 1
        return count
    
    def delta_for_flip(coloring, pos):
        """Compute change in mono count if we flip position pos."""
        new_val = 1 - coloring[pos]
        old_val = coloring[pos]
        delta = 0
        for ap_idx, others in pos_aps[pos]:
            ap = ap_list[ap_idx]
            # Before flip: is this AP mono?
            was_mono = all(coloring[x] == old_val for x in ap)
            # After flip: pos becomes new_val
            # Is it mono after? All 4 must be new_val
            would_be_mono = (coloring[others[0]] == new_val and 
                           coloring[others[1]] == new_val and 
                           coloring[others[2]] == new_val)
            if was_mono and not would_be_mono:
                delta -= 1
            elif not was_mono and would_be_mono:
                delta += 1
        return delta
    
    def greedy_polish(coloring, max_iterations=10000):
        """Greedy local search: keep flipping if it reduces count."""
        count = count_mono(coloring)
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            if time.time() > deadline - 5:
                break
            improved = False
            iters += 1
            # Try each position
            for pos in range(n):
                delta = delta_for_flip(coloring, pos)
                if delta < 0:
                    coloring[pos] = 1 - coloring[pos]
                    count += delta
                    improved = True
        return coloring, count
    
    # Generate initial colorings
    def thue_morse():
        colors = []
        for i in range(n):
            cnt = bin(i).count('1')
            colors.append(cnt % 2)
        return colors
    
    def random_coloring(rng):
        return rng.integers(0, 2, n).tolist()
    
    def periodic_coloring(period, offset=0):
        """Create a periodic coloring with given period."""
        colors = []
        for i in range(n):
            colors.append((i + offset) % period % 2)
        return colors
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    startings = []
    
    # Thue-Morse
    startings.append(thue_morse())
    
    # Various periodic patterns
    for period in [2, 3, 4, 5, 6, 7, 8]:
        for offset in range(period):
            startings.append(periodic_coloring(period, offset))
    
    # Random starts
    for _ in range(20):
        startings.append(random_coloring(rng))
    
    # Also try Thue-Morse with perturbations
    tm = thue_morse()
    for _ in range(10):
        perturbed = tm[:]
        for _ in range(rng.integers(3, 15)):
            bit = rng.integers(0, n)
            perturbed[bit] = 1 - perturbed[bit]
        startings.append(perturbed)
    
    best_coloring = None
    best_count = float('inf')
    
    # Greedy polish all starting points
    for start in startings:
        if time.time() > deadline - 10:
            break
        coloring, count = greedy_polish(start[:])
        if count < best_count:
            best_count = count
            best_coloring = coloring[:]
    
    # Simulated annealing from best found
    current_coloring = best_coloring[:]
    current_count = best_count
    
    # Multiple SA runs with different temperatures
    for sa_run in range(30):
        if time.time() > deadline - 5:
            break
        
        if sa_run == 0:
            current_coloring = best_coloring[:]
            current_count = best_count
        elif sa_run % 2 == 0:
            # Perturb best
            current_coloring = best_coloring[:]
            num_flips = rng.integers(3, 12)
            for _ in range(num_flips):
                bit = rng.integers(0, n)
                current_coloring[bit] = 1 - current_coloring[bit]
            current_count = count_mono(current_coloring)
        else:
            # Random restart
            current_coloring = random_coloring(rng)
            current_count = count_mono(current_coloring)
        
        # Simulated annealing
        temp = 30.0
        cooling = 0.997
        min_temp = 0.5
        steps_per_temp = 50
        
        while temp > min_temp:
            if time.time() > deadline - 5:
                break
            
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5:
                    break
                
                pos = rng.integers(0, n)
                delta = delta_for_flip(current_coloring, pos)
                
                if delta < 0:
                    current_coloring[pos] = 1 - current_coloring[pos]
                    current_count += delta
                elif delta == 0:
                    if rng.random() < 0.3:
                        current_coloring[pos] = 1 - current_coloring[pos]
                elif delta > 0:
                    if rng.random() < np.exp(-delta / temp):
                        current_coloring[pos] = 1 - current_coloring[pos]
                        current_count += delta
                
                temp *= cooling
            
            if current_count < best_count:
                best_count = current_count
                best_coloring = current_coloring[:]
        
        # Final greedy polish
        current_coloring, current_count = greedy_polish(current_coloring[:], max_iterations=5000)
        if current_count < best_count:
            best_count = current_count
            best_coloring = current_coloring[:]
    
    return best_coloring
