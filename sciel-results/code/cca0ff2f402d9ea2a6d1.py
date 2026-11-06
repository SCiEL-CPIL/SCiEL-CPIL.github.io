import random
import time

def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        
    k4_edge_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    verts = [i, j, k, l]
                    e1 = edge_idx[(verts[0], verts[1])]
                    e2 = edge_idx[(verts[0], verts[2])]
                    e3 = edge_idx[(verts[0], verts[3])]
                    e4 = edge_idx[(verts[1], verts[2])]
                    e5 = edge_idx[(verts[1], verts[3])]
                    e6 = edge_idx[(verts[2], verts[3])]
                    k4_edge_indices.append([e1, e2, e3, e4, e5, e6])
                    
    num_k4s = len(k4_edge_indices)
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def count_colors_for_k4(k4_idx, colours):
        e_indices = k4_edge_indices[k4_idx]
        c0 = colours[e_indices[0]]
        c1 = colours[e_indices[1]]
        c2 = colours[e_indices[2]]
        c3 = colours[e_indices[3]]
        c4 = colours[e_indices[4]]
        c5 = colours[e_indices[5]]
        # Check if at least 5 distinct
        # If any two are equal, count might be < 6.
        # We need len(set) >= 5.
        # This is equivalent to NOT (at least 2 pairs equal OR 3 same etc)
        # Simpler to just use set for 6 elements.
        return len({c0, c1, c2, c3, c4, c5})
        
    def get_cost(colours):
        cost = 0
        for k4_idx in range(num_k4s):
            if count_colors_for_k4(k4_idx, colours) < 5:
                cost += 1
        return cost
        
    def local_search(K, time_limit):
        colours = [random.randint(0, K - 1) for _ in range(num_edges)]
        cost = get_cost(colours)
        best_cost = cost
        best_colours = colours[:]
        
        start_time = time.time()
        current_temp = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if cost == 0:
                return colours
            
            e_idx = random.randint(0, num_edges - 1)
            old_colour = colours[e_idx]
            
            if K > 1:
                new_colour = random.randint(0, K - 1)
                if new_colour == old_colour:
                    continue
            else:
                continue
                
            colours[e_idx] = new_colour
            
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_count = count_colors_for_k4(k4_idx, colours) # Wait, colours is updated
                # We need old count BEFORE update.
                # So revert, get old, update, get new.
                pass
            
            # Corrected delta calculation
            colours[e_idx] = old_colour
            old_counts = [count_colors_for_k4(k4_idx, colours) for k4_idx in edge_to_k4s[e_idx]]
            
            colours[e_idx] = new_colour
            new_counts = [count_colors_for_k4(k4_idx, colours) for k4_idx in edge_to_k4s[e_idx]]
            
            for old_c, new_c in zip(old_counts, new_counts):
                if old_c < 5 and new_c >= 5:
                    delta -= 1
                elif old_c >= 5 and new_c < 5:
                    delta += 1
                    
            new_cost = cost + delta
            
            if new_cost < cost or random.random() < pow(2.718, -delta / current_temp):
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_colours = colours[:]
            else:
                colours[e_idx] = old_colour
                
            current_temp *= cooling
            if current_temp < 0.01:
                current_temp = 0.01
                
        return best_colours if best_cost == 0 else None

    start_total = time.time()
    for K in [13, 14, 15]:
        if time.time() - start_total > 135:
            break
        result = local_search(K, 40)
        if result is not None:
            if get_cost(result) == 0:
                return result

    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
