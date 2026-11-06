import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    for target_n in [26, 27, 28]:
        if time.time() > deadline - 20: break
        
        n = target_n
        # Greedy Initialization
        colors = [[0]*n for _ in range(n)]
        masks = [[0]*n for _ in range(5)]
        
        # We build the graph vertex by vertex
        # For each new vertex v, we assign colors to (u, v) for u < v
        # To minimize energy, we pick the color that adds the least new C4s
        
        # Helper to calculate delta for adding edge (u, v) with color c
        # Note: masks are currently for vertices 0..v-1
        # We are adding vertex v, so masks for v are initially 0.
        # We need to check interaction with existing vertices.
        
        # Actually, simpler: just use the same delta logic as SA,
        # but we are "adding" edges one by one.
        
        # Let's just use Random Init + SA, it might be simpler and robust enough.
        # Greedy init is good but complex to implement correctly with the delta logic
        # because the graph is incomplete.
        
        # Let's try Random Init + SA with many restarts.
        
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5: break
            
            # Random Init
            colors = [[0]*n for _ in range(n)]
            masks = [[0]*n for _ in range(5)]
            for i in range(n):
                for j in range(i+1, n):
                    c = random.randint(0, 4)
                    colors[i][j] = c
                    colors[j][i] = c
                    masks[c][i] |= (1 << j)
                    masks[c][j] |= (1 << i)
                    
            # Calculate Initial Energy
            def calc_energy(masks, n):
                e = 0
                for c in range(5):
                    for i in range(n):
                        mi = masks[c][i]
                        for j in range(i+1, n):
                            k = (mi & masks[c][j]).bit_count()
                            if k >= 2:
                                e += k * (k - 1) // 2
                return e
                
            current_energy = calc_energy(masks, n)
            
            if current_energy == 0:
                found = True
                break
                
            # Simulated Annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            steps = 0
            max_steps = 50000 # Adjust based on time
            
            while T > min_T and time.time() < deadline - 2:
                # Pick random edge
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j: continue
                if i > j: i, j = j, i
                
                c_old = colors[i][j]
                c_new = random.randint(0, 4)
                if c_old == c_new: continue
                
                # Calculate Delta
                delta = 0
                # Affected pairs: (i, k) and (j, k) for k != i, j
                
                # Pre-fetch masks for speed
                M_cold_i = masks[c_old][i]
                M_cold_j = masks[c_old][j]
                M_new_i = masks[c_new][i]
                M_new_j = masks[c_new][j]
                
                # We need to iterate k
                # Optimization: Only iterate k if there are potential changes
                # But for n=26, full iteration is fine.
                
                for k in range(n):
                    if k == i or k == j: continue
                    
                    # Pair (i, k)
                    # c_old: if (j,k) in c_old, count decreases
                    if (M_cold_j >> k) & 1:
                        N_old = (M_cold_i & masks[c_old][k]).bit_count()
                        delta += 1 - N_old
                        
                    # c_new: if (j,k) in c_new, count increases
                    if (M_new_j >> k) & 1:
                        N_old = (M_new_i & masks[c_new][k]).bit_count()
                        delta += N_old
                        
                    # Pair (j, k)
                    # c_old: if (i,k) in c_old, count decreases
                    if (M_cold_i >> k) & 1:
                        N_old = (M_cold_j & masks[c_old][k]).bit_count()
                        delta += 1 - N_old
                        
                    # c_new: if (i,k) in c_new, count increases
                    if (M_new_i >> k) & 1:
                        N_old = (M_new_j & masks[c_new][k]).bit_count()
                        delta += N_old
                        
                # Metropolis
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    colors[i][j] = c_new
                    colors[j][i] = c_new
                    masks[c_old][i] &= ~(1 << j)
                    masks[c_old][j] &= ~(1 << i)
                    masks[c_new][i] |= (1 << j)
                    masks[c_new][j] |= (1 << i)
                    current_energy += delta
                    
                    if current_energy == 0:
                        found = True
                        break
                        
                T *= cooling
                steps += 1
                
            if found:
                break
                
        if found:
            best_n = n
            flat_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    flat_colors.append(colors[i][j])
            best_colors = flat_colors
            # If we found 26, we might try 27 if time permits
            # But 27 is very hard.
            if n == 26 and time.time() < deadline - 30:
                continue
            else:
                break
                
    return {'n': best_n, 'colors': best_colors}
