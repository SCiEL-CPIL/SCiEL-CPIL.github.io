import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= n
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
            
    total_aps = len(aps)
    
    # Precompute neighbors for fast delta calculation
    # neighbors[p] is a list of tuples (idx1, idx2, idx3) for each AP containing p
    # where idx1, idx2, idx3 are the other 3 indices in that AP
    neighbors = [[] for _ in range(n)]
    
    for ap in aps:
        i1, i2, i3, i4 = ap
        # For i1, others are i2, i3, i4
        neighbors[i1].append((i2, i3, i4))
        # For i2, others are i1, i3, i4
        neighbors[i2].append((i1, i3, i4))
        # For i3, others are i1, i2, i4
        neighbors[i3].append((i1, i2, i4))
        # For i4, others are i1, i2, i3
        neighbors[i4].append((i1, i2, i3))

    def count_mono(coloring):
        count = 0
        for i1, i2, i3, i4 in aps:
            if coloring[i1] == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count

    def calc_delta(p, coloring):
        """
        Calculate the change in mono-AP count if we flip position p.
        """
        current_bit = coloring[p]
        delta = 0
        for o1, o2, o3 in neighbors[p]:
            c1 = coloring[o1]
            c2 = coloring[o2]
            c3 = coloring[o3]
            if c1 == c2 == c3:
                if c1 == current_bit:
                    # Flipping p breaks this mono AP
                    delta -= 1
                else:
                    # Flipping p creates this mono AP
                    delta += 1
        return delta

    best_coloring = None
    best_score = float('inf')
    
    # Simulated Annealing parameters
    T_initial = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    while time.time() < limit:
        # Initialize random coloring
        coloring = np.random.randint(0, 2, n)
        current_score = count_mono(coloring)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = coloring.copy()
            
        T = T_initial
        
        # Inner SA loop
        while T > T_min and time.time() < limit:
            p = random.randint(0, n - 1)
            delta = calc_delta(p, coloring)
            
            # Accept if improvement, or with probability exp(-delta/T) if worsening
            if delta < 0 or random.random() < np.exp(-delta / T):
                coloring[p] = 1 - coloring[p]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            
            T *= cooling_rate
            
            # Check time periodically to avoid long overhead
            if random.randint(0, 100) == 0:
                if time.time() > limit - 5:
                    break

    if best_coloring is None:
         # Fallback to random or simple pattern
         return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
