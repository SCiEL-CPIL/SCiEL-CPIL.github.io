import random
import time
import math

def construct():
    n = 26
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Use bitmasks for adjacency to speed up common neighbor calculation
    # adj[c][u] is an integer bitmask of neighbors of u in color c
    # n=26 fits in an integer.
    
    def init_adj(colors):
        adj = [[0] * n for _ in range(5)]
        for k, (u, v) in enumerate(edges):
            c = colors[k]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        return adj

    def get_common_count(adj, u, v, c):
        return bin(adj[c][u] & adj[c][v]).count('1')

    def calculate_energy(adj):
        energy = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cnt = get_common_count(adj, u, v, c)
                    if cnt >= 2:
                        energy += cnt * (cnt - 1) // 2
        return energy

    # Initialize
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    adj = init_adj(colors)
    current_energy = calculate_energy(adj)
    
    start_time = time.time()
    timeout = 110
    T = 1.0
    cooling_rate = 0.9995 # Slower cooling for better exploration
    
    # Simulated Annealing
    while current_energy > 0 and time.time() - start_time < timeout:
        if T < 0.01:
            T = 0.01
            
        # Pick a random edge
        k = random.randint(0, num_edges - 1)
        u, v = edges[k]
        old_color = colors[k]
        
        # Pick a new color different from old
        new_color = random.randint(0, 4)
        while new_color == old_color:
            new_color = random.randint(0, 4)
            
        # Calculate delta energy
        # Remove (u,v) from old_color: energy change is -C(cnt_old, 2)
        cnt_old = get_common_count(adj, u, v, old_color)
        delta_remove = - (cnt_old * (cnt_old - 1) // 2)
        
        # Add (u,v) to new_color: energy change is +C(cnt_new, 2)
        cnt_new = get_common_count(adj, u, v, new_color)
        delta_add = cnt_new * (cnt_new - 1) // 2
        
        delta_E = delta_remove + delta_add
        
        # Acceptance criterion
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept move
            colors[k] = new_color
            
            # Update adjacency structures
            adj[old_color][u] &= ~(1 << v)
            adj[old_color][v] &= ~(1 << u)
            adj[new_color][u] |= (1 << v)
            adj[new_color][v] |= (1 << u)
            
            current_energy += delta_E
            
        # Cool down
        T *= cooling_rate
        
        if current_energy == 0:
            break

    if current_energy == 0:
        return {'n': n, 'colors': colors}
    
    # Fallback to n=11
    n_seed = 11
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
