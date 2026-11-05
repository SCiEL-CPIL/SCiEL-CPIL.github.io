import random
import time
import math

def construct():
    # Pre-compute K4s and their edge indices
    edge_index = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_index[(i, j)] = idx
            idx += 1
    total_edges = 120
    
    # List of K4s as tuples of 4 vertices
    k4s = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    k4s.append((i, j, k, l))
                    
    # For each edge, store list of (k4_index, other_edges_list)
    edge_k4_info = [[] for _ in range(total_edges)]
    
    for k4_idx, k4 in enumerate(k4s):
        i, j, k, l = k4
        edges = [
            edge_index[(i, j)], edge_index[(i, k)], edge_index[(i, l)],
            edge_index[(j, k)], edge_index[(j, l)], edge_index[(k, l)]
        ]
        for e in edges:
            others = [x for x in edges if x != e]
            edge_k4_info[e].append((k4_idx, others))

    def get_energy(colours):
        bad = 0
        for k4 in k4s:
            i, j, k, l = k4
            edges = [
                edge_index[(i, j)], edge_index[(i, k)], edge_index[(i, l)],
                edge_index[(j, k)], edge_index[(j, l)], edge_index[(k, l)]
            ]
            # Count distinct colors
            c = set()
            for e in edges:
                c.add(colours[e])
            if len(c) < 5:
                bad += 1
        return bad

    def delta_energy(colours, edge_idx, new_colour):
        old_colour = colours[edge_idx]
        if old_colour == new_colour:
            return 0
            
        delta = 0
        for k4_idx, others in edge_k4_info[edge_idx]:
            others_colors = [colours[e] for e in others]
            
            old_set = set(others_colors)
            old_set.add(old_colour)
            old_count = len(old_set)
            
            new_set = set(others_colors)
            new_set.add(new_colour)
            new_count = len(new_set)
            
            old_bad = 1 if old_count < 5 else 0
            new_bad = 1 if new_count < 5 else 0
            
            delta += (new_bad - old_bad)
            
        return delta

    def sa_search(k, max_iters=20000, t_start=1.0, t_end=0.01):
        colours = [random.randint(0, k-1) for _ in range(total_edges)]
        current_energy = get_energy(colours)
        best_colours = colours[:]
        best_energy = current_energy
        
        if current_energy == 0:
            return colours
            
        for i in range(max_iters):
            t = t_start * (t_end / t_start) ** (i / max_iters)
            
            e = random.randint(0, total_edges - 1)
            new_c = random.randint(0, k-1)
            
            d = delta_energy(colours, e, new_c)
            
            if d < 0:
                colours[e] = new_c
                current_energy += d
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colours = colours[:]
                if current_energy == 0:
                    return colours
            else:
                if random.random() < math.exp(-d / t):
                    colours[e] = new_c
                    current_energy += d
                    
        return best_colours

    total_start = time.time()
    
    # Try k = 13, 14, 15, 16
    for k in [13, 14, 15, 16]:
        if time.time() - total_start > 100:
            break
            
        # Run SA search
        # We can do multiple restarts if time permits
        # Let's do 1 run of 50000 iters
        sol = sa_search(k, max_iters=50000)
        
        if get_energy(sol) == 0:
            return sol
            
    # Fallback: Use the seed logic (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
