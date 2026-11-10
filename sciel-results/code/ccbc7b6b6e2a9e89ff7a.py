import numpy as np
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] contains tuples (j, k, l) for each AP containing i
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))

    def count_mono(coloring):
        count = 0
        for i1, i2, i3, i4 in aps:
            if coloring[i1] == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count

    # Initial State: Random
    coloring = np.random.randint(0, 2, N)
    current_energy = count_mono(coloring)
    best_energy = current_energy
    best_coloring = coloring.copy()
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    # Main Loop
    while time.time() < deadline and T > min_T:
        p = random.randint(0, N - 1)
        
        delta = 0
        cp = coloring[p]
        
        for x, y, z in neighbors[p]:
            cx = coloring[x]
            cy = coloring[y]
            cz = coloring[z]
            
            # Check if x,y,z match cp
            b1 = (cx == cp)
            b2 = (cy == cp)
            b3 = (cz == cp)
            
            # Before flip: mono if all match cp
            before = 1 if (b1 and b2 and b3) else 0
            # After flip: p becomes 1-cp. Mono if all x,y,z match 1-cp (i.e., none match cp)
            after = 1 if (not b1 and not b2 and not b3) else 0
            
            delta += after - before
            
        # Acceptance criterion
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            accept = random.random() < math.exp(-delta / T)
            
        if accept:
            coloring[p] = 1 - coloring[p]
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_coloring = coloring.copy()
                
        T *= cooling_rate
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
