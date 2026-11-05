import random
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

# Precompute edges
edges = []
edge_to_u_v = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))
        edge_to_u_v[idx] = (i, j)
        idx += 1

# Precompute 4-cycles
# A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific pairing of edges.
# There are 3 distinct 4-cycles for each set of 4 vertices.
# We store the 4 edge indices for each cycle.
cycles = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

# Map vertex pair to edge index
pair_to_edge = {}
for i, (u, v) in enumerate(edges):
    pair_to_edge[(u, v)] = i
    pair_to_edge[(v, u)] = i

for verts in combinations(range(N), 4):
    a, b, c, d = verts
    # 3 cycles:
    # 1: a-b, b-c, c-d, d-a
    e1 = pair_to_edge[(a, b)]
    e2 = pair_to_edge[(b, c)]
    e3 = pair_to_edge[(c, d)]
    e4 = pair_to_edge[(d, a)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)
        
    # 2: a-c, c-b, b-d, d-a  -> a-c, c-b, b-d, d-a
    # Edges: (a,c), (b,c), (b,d), (a,d)
    e1 = pair_to_edge[(a, c)]
    e2 = pair_to_edge[(b, c)]
    e3 = pair_to_edge[(b, d)]
    e4 = pair_to_edge[(a, d)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)
        
    # 3: a-b, b-d, d-c, c-a
    # Edges: (a,b), (b,d), (c,d), (a,c)
    e1 = pair_to_edge[(a, b)]
    e2 = pair_to_edge[(b, d)]
    e3 = pair_to_edge[(c, d)]
    e4 = pair_to_edge[(a, c)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_solution = [0] * NUM_EDGES
    best_score = 0
    
    # Initial simple solution to ensure we have something valid
    # Use the triangle decomposition strategy as a fallback/initial
    # Partition into 5 triangles + 1 vertex
    # T1: 0,1,2; T2: 3,4,5; T3: 6,7,8; T4: 9,10,11; T5: 12,13,14; V15: 15
    # Internal edges: unique colors
    # External edges: color 100
    
    trial_colors = [-1] * NUM_EDGES
    used_colors = set()
    
    # Assign internal triangle colors
    tri_groups = [(0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14)]
    for i, (u,v,w) in enumerate(tri_groups):
        c = i
        trial_colors[pair_to_edge[(u,v)]] = c
        trial_colors[pair_to_edge[(v,w)]] = c
        trial_colors[pair_to_edge[(u,w)]] = c
        used_colors.add(c)
        
    # Assign external colors
    bg_color = 100
    used_colors.add(bg_color)
    for i in range(NUM_EDGES):
        if trial_colors[i] == -1:
            trial_colors[i] = bg_color
            
    best_solution = trial_colors[:]
    best_score = len(used_colors)
    
    # Greedy Search
    while time.time() < deadline:
        random.seed() # Or just let it run
        edge_order = random.sample(range(NUM_EDGES), NUM_EDGES)
        colors = [-1] * NUM_EDGES
        used_colors = set()
        success = True
        
        for e_idx in edge_order:
            # Determine allowed colors
            Allowed = set(range(MAX_COLORS))
            
            for c_idx in edge_to_cycles[e_idx]:
                cyc = cycles[c_idx]
                # Get the other 3 edges
                others = [e for e in cyc if e != e_idx]
                c1 = colors[others[0]]
                c2 = colors[others[1]]
                c3 = colors[others[2]]
                
                # If any is uncolored, this cycle doesn't constrain us yet
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue
                    
                S = {c1, c2, c3}
                if len(S) == 3:
                    Allowed = Allowed.intersection(S)
                    
            if not Allowed:
                success = False
                break
                
            # Prefer new colors
            NewAllowed = Allowed.difference(used_colors)
            if NewAllowed:
                # Pick randomly from new colors to maximize diversity
                c = random.choice(list(NewAllowed))
            else:
                c = random.choice(list(Allowed))
                
            colors[e_idx] = c
            used_colors.add(c)
            
        if success:
            num_used = len(used_colors)
            if num_used > best_score:
                best_score = num_used
                best_solution = colors[:]
                
    return best_solution
