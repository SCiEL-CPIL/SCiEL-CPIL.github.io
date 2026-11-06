import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                adj_i = adj[c][i]
                for j in range(i + 1, n):
                    common = adj_i & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def greedy_init(n):
        """
        Greedy construction: assign colors to edges one by one,
        choosing the color that minimizes the number of C4s formed.
        """
        colors = [0] * (n * (n - 1) // 2)
        adj = [[0] * n for _ in range(5)]
        
        for i in range(n):
            for j in range(i + 1, n):
                ei = edge_idx(i, j, n)
                best_c = 0
                best_cost = float('inf')
                
                for c in range(5):
                    # Calculate cost of adding (i,j) to color c
                    cost = 0
                    # A C4 is formed if there is a vertex w such that
                    # (i,w) is in c AND (j,w) is in c.
                    # This is equivalent to counting common neighbors of i and j in adj[c].
                    # Wait, if we add (i,j), a C4 is formed with any w where
                    # i-w and j-w are already in color c.
                    # So cost = number of w such that (i,w) in c and (j,w) in c.
                    # This is (adj[c][i] & adj[c][j]).bit_count()
                    
                    common = (adj[c][i] & adj[c][j]).bit_count()
                    if common < best_cost:
                        best_cost = common
                        best_c = c
                    
                    # Tie-breaking: randomize if equal? Or prefer 0?
                    # Let's add a small random perturbation to break ties
                    # to allow different greedy paths.
                    if common == best_cost and random.random() < 0.1:
                         best_c = c
                         best_cost = common + 0.1 # slightly worse to allow change? 
                         # Actually, just picking the first one is fine for deterministic greedy.
                         # Let's stick to first min for speed and determinism.
                
                colors[ei] = best_c
                adj[best_c][i] |= (1 << j)
                adj[best_c][j] |= (1 << i)
                
        return colors, adj

    def polish(adj, colors, n, edge_list, energy):
        """
        Deterministic hill climbing.
        """
        improved = True
        max_passes = 100
        for _ in range(max_passes):
            if not improved:
                break
            if time_left() < 1:
                break
            
            best_delta = 0
            best_ei = -1
            best_c_new = -1
            
            num_edges = len(colors)
            for ei in range(num_edges):
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                
                for c_new in range(5):
                    if c_new == c_old:
                        continue
                    
                    delta = 0
                    
                    # Remove from c_old
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_old][w]
                        if mask_w & (1 << v):
                            k = (adj[c_old][u] & mask_w).bit_count()
                            delta -= (k - 1)
                        if mask_w & (1 << u):
                            k = (adj[c_old][v] & mask_w).bit_count()
                            delta -= (k - 1)
                    
                    # Add to c_new
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_new][w]
                        if mask_w & (1 << v):
                            k = (adj[c_new][u] & mask_w).bit_count()
                            delta += k
                        if mask_w & (1 << u):
                            k = (adj[c_new][v] & mask_w).bit_count()
                            delta += k
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_ei = ei
                        best_c_new = c_new
            
            if best_ei != -1:
                u, v, e_idx = edge_list[best_ei]
                c_old = colors[e_idx]
                c_new = best_c_new
                
                energy += best_delta
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                improved = True
            else:
                improved = False
        
        return energy, colors, adj, improved

    best_n = 1
    best_colors = [] # Fixed: was [0]
    
    # Try n=26 first
    target_ns = [26]
    
    for target_n in target_ns:
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        max_restarts = 20 # Reduced restarts, relying on better init
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            # Use Greedy Init
            colors, adj = greedy_init(n)
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            best_adj_local = [row[:] for row in adj]
            
            # SA Phase
            temp = 1.0
            cooling = 0.999
            min_temp = 0.001
            max_steps = 20000
            stuck_count = 0
            max_stuck = 2000
            
            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                ei = random.randint(0, num_edges - 1)
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # Remove from c_old
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj[c_old][w]
                    if mask_w & (1 << v):
                        k = (adj[c_old][u] & mask_w).bit_count()
                        delta -= (k - 1)
                    if mask_w & (1 << u):
                        k = (adj[c_old][v] & mask_w).bit_count()
                        delta -= (k - 1)
                
                # Add to c_new
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj[c_new][w]
                    if mask_w & (1 << v):
                        k = (adj[c_new][u] & mask_w).bit_count()
                        delta += k
                    if mask_w & (1 << u):
                        k = (adj[c_new][v] & mask_w).bit_count()
                        delta += k
                
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        best_adj_local = [row[:] for row in adj]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break
            
            # Polish Phase
            if time_left() > 2:
                energy, colors, adj, improved = polish(adj, colors, n, edge_list, energy)
                if energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                    best_adj_local = [row[:] for row in adj]
            
            # If we got very close, maybe try again from this state?
            # For now, just restart.
            
        if found:
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
