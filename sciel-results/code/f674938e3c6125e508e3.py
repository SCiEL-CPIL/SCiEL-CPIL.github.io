import random

N = 16
NUM_EDGES = N * (N - 1) // 2

edge_to_idx = {}
idx_to_edge = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        idx_to_edge.append((i, j))
        idx += 1

edge_cycles = [[] for _ in range(NUM_EDGES)]

for u in range(N):
    for v in range(u + 1, N):
        for w in range(v + 1, N):
            for x in range(w + 1, N):
                e_uv = edge_to_idx[(u, v)]
                e_vw = edge_to_idx[(v, w)]
                e_wx = edge_to_idx[(w, x)]
                e_xu = edge_to_idx[(x, u)]
                
                edge_cycles[e_uv].append((e_vw, e_wx, e_xu))
                edge_cycles[e_vw].append((e_wx, e_xu, e_uv))
                edge_cycles[e_wx].append((e_xu, e_uv, e_vw))
                edge_cycles[e_xu].append((e_uv, e_vw, e_wx))
                
                e_vx = edge_to_idx[(v, x)]
                e_uw = edge_to_idx[(u, w)]
                
                edge_cycles[e_uv].append((e_vx, e_wx, e_uw))
                edge_cycles[e_vx].append((e_wx, e_uw, e_uv))
                edge_cycles[e_wx].append((e_uw, e_uv, e_vx))
                edge_cycles[e_uw].append((e_uv, e_vx, e_wx))
                
                edge_cycles[e_uw].append((e_vw, e_vx, e_xu))
                edge_cycles[e_vw].append((e_vx, e_xu, e_uw))
                edge_cycles[e_vx].append((e_xu, e_uw, e_vw))
                edge_cycles[e_xu].append((e_uw, e_vw, e_vx))

def can_assign_new_color(colors, edge_idx):
    for e1, e2, e3 in edge_cycles[edge_idx]:
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        if c1 != c2 and c2 != c3 and c1 != c3:
            return False
    return True

def construct():
    best_score = 0
    best_colors = [0] * NUM_EDGES
    
    num_restarts = 5000
    
    for _ in range(num_restarts):
        colors = [0] * NUM_EDGES
        order = list(range(NUM_EDGES))
        random.shuffle(order)
        
        next_color = 0
        for e in order:
            if next_color <= 119 and can_assign_new_color(colors, e):
                colors[e] = next_color
                next_color += 1
        
        score = len(set(colors))
        if score > best_score:
            best_score = score
            best_colors = colors.copy()
            
    return best_colors
