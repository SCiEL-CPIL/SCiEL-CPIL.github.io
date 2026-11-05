import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    edge_list = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edge_list)
            edge_list.append((i, j))
            edge_to_idx[(i, j)] = idx
    
    num_edges = len(edge_list)
    
    # Precompute all K4s and their edges
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
    k4_edges = []
    for k4 in k4s:
        edges_in_k4 = []
        for a, b in combinations(k4, 2):
            edges_in_k4.append(edge_to_idx[(a, b)])
        k4_edges.append(edges_in_k4)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    # For each K4, track color counts for quick violation check
    # k4_color_counts[ki] = dict color -> count
    
    def init_color_counts(colors):
        counts = [{} for _ in range(num_k4s)]
        for ki, edges in enumerate(k4_edges):
            cdict = {}
            for e in edges:
                c = colors[e]
                cdict[c] = cdict.get(c, 0) + 1
            counts[ki] = cdict
        return counts
    
    def count_violations_from_counts(counts):
        viol = 0
        for cdict in counts:
            if len(cdict) < 5:
                viol += 1
        return viol
    
    best_result = None
    best_num_colors = 999
    
    for num_colors in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
            
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
            counts = init_color_counts(colors)
            cur_viol = count_violations_from_counts(counts)
            
            if cur_viol == 0:
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_result = colors[:]
                break
            
            # Local search with incremental updates
            max_iters = 100000
            no_improve = 0
            max_no_improve = 5000
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if no_improve > max_no_improve:
                    break
                
                # Find a violating K4
                viol_ki = -1
                for ki in range(num_k4s):
                    if len(counts[ki]) < 5:
                        viol_ki = ki
                        break
                
                if viol_ki == -1:
                    cur_viol = 0
                    break
                
                edges = k4_edges[viol_ki]
                cset = set(counts[viol_ki].keys())
                
                # Try recoloring edges in this K4
                shuffled = edges[:]
                random.shuffle(shuffled)
                
                found = False
                for e in shuffled:
                    old_c = colors[e]
                    # Try colors not in current K4's color set
                    available = [c for c in range(num_colors) if c not in cset]
                    if not available:
                        continue
                    new_c = random.choice(available)
                    
                    # Update: change edge e from old_c to new_c
                    # Affects all K4s containing e
                    for ki2 in edge_to_k4s[e]:
                        # Remove old_c
                        cd = counts[ki2]
                        cd[old_c] -= 1
                        if cd[old_c] == 0:
                            del cd[old_c]
                        # Add new_c
                        cd[new_c] = cd.get(new_c, 0) + 1
                    
                    new_viol = count_violations_from_counts(counts)
                    
                    if new_viol < cur_viol:
                        colors[e] = new_c
                        cur_viol = new_viol
                        no_improve = 0
                        found = True
                        break
                    else:
                        # Revert
                        for ki2 in edge_to_k4s[e]:
                            cd = counts[ki2]
                            cd[new_c] -= 1
                            if cd[new_c] == 0:
                                del cd[new_c]
                            cd[old_c] = cd.get(old_c, 0) + 1
                
                if not found:
                    no_improve += 1
                    # Random perturbation
                    if no_improve % 100 == 0:
                        e = random.randrange(num_edges)
                        old_c = colors[e]
                        new_c = random.randint(0, num_colors - 1)
                        if new_c != old_c:
                            for ki2 in edge_to_k4s[e]:
                                cd = counts[ki2]
                                cd[old_c] -= 1
                                if cd[old_c] == 0:
                                    del cd[old_c]
                                cd[new_c] = cd.get(new_c, 0) + 1
                            colors[e] = new_c
                            cur_viol = count_violations_from_counts(counts)
            
            if cur_viol == 0:
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_result = colors[:]
                break
    
    if best_result is not None:
        return best_result
    
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
