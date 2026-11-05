import numpy as np

N = 16

def construct():
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edges.append((u, v))
            
    num_edges = len(edges)
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    colors = [-1] * num_edges
    next_color = 0
    color_counts = [0] * 120 # Max 120 colors

    cycles_for_edge = [[] for _ in range(num_edges)]
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for idx_x in range(len(others)):
            x = others[idx_x]
            for idx_y in range(idx_x + 1, len(others)):
                y = others[idx_y]
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                e1 = edge_to_idx[(v, y)]
                e2 = edge_to_idx[(y, x)]
                e3 = edge_to_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))

    for i in range(num_edges):
        u, v = edges[i]
        
        allowed_set = None
        for (e1, e2, e3) in cycles_for_edge[i]:
            if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                s = {colors[e1], colors[e2], colors[e3]}
                if len(s) == 3:
                    if allowed_set is None:
                        allowed_set = s
                    else:
                        allowed_set = allowed_set.intersection(s)
        
        if allowed_set is not None:
            # Pick the least used color in allowed_set
            best_c = -1
            min_count = float('inf')
            for c in allowed_set:
                if color_counts[c] < min_count:
                    min_count = color_counts[c]
                    best_c = c
            colors[i] = best_c
            color_counts[best_c] += 1
        else:
            colors[i] = next_color
            color_counts[next_color] = 1
            next_color += 1
            
    return colors
