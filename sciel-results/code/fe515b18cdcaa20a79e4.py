import itertools
import time

N = 16
EDGES = []
for i in range(N):
    for j in range(i+1, N):
        EDGES.append((i, j))

edge_index = {}
for idx, (u, v) in enumerate(EDGES):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx

# Precompute all 4-cycles
# A 4-cycle is defined by 4 distinct vertices {a, b, c, d}.
# There are 3 distinct 4-cycles on these vertices:
# (a,b,c,d), (a,b,d,c), (a,c,b,d)
# Edges:
# C1: (a,b), (b,c), (c,d), (d,a)
# C2: (a,b), (b,d), (d,c), (c,a)
# C3: (a,c), (c,b), (b,d), (d,a)

cycles = []
# To avoid duplicates, we can just iterate over combinations of 4 vertices.
# For each combination, add the 3 cycles.
for verts in itertools.combinations(range(N), 4):
    a, b, c, d = verts
    # Cycle 1: a-b-c-d-a
    e1 = edge_index[(a, b)]
    e2 = edge_index[(b, c)]
    e3 = edge_index[(c, d)]
    e4 = edge_index[(d, a)]
    cycles.append((e1, e2, e3, e4))
    
    # Cycle 2: a-b-d-c-a
    e1 = edge_index[(a, b)]
    e2 = edge_index[(b, d)]
    e3 = edge_index[(d, c)]
    e4 = edge_index[(c, a)]
    cycles.append((e1, e2, e3, e4))
    
    # Cycle 3: a-c-b-d-a
    e1 = edge_index[(a, c)]
    e2 = edge_index[(c, b)]
    e3 = edge_index[(b, d)]
    e4 = edge_index[(d, a)]
    cycles.append((e1, e2, e3, e4))

num_cycles = len(cycles)
num_edges = len(EDGES)

# Map pair of edge indices to list of cycle indices
# pair_key = (min(e1, e2), max(e1, e2))
cycles_for_pair = {}
for i, cycle in enumerate(cycles):
    for e1, e2 in itertools.combinations(cycle, 2):
        key = (min(e1, e2), max(e1, e2))
        if key not in cycles_for_pair:
            cycles_for_pair[key] = []
        cycles_for_pair[key].append(i)

all_pairs = list(cycles_for_pair.keys())

def construct():
    # Greedy Set Cover on pairs
    active = [True] * num_cycles
    merges = []
    
    # Optimization: We can precompute the length of lists, but they are small.
    # We need to find the pair with max active cycles count.
    
    while True:
        # Check if any active
        if not any(active):
            break
            
        max_count = 0
        best_pair = None
        
        for pair in all_pairs:
            # Count active cycles for this pair
            cnt = 0
            for c_idx in cycles_for_pair[pair]:
                if active[c_idx]:
                    cnt += 1
            if cnt > max_count:
                max_count = cnt
                best_pair = pair
                
        if max_count == 0:
            # Should not happen if any(active) is True, unless logic error
            break
            
        merges.append(best_pair)
        
        # Deactivate cycles covered by this pair
        for c_idx in cycles_for_pair[best_pair]:
            active[c_idx] = False
            
    # Construct colors
    colors = list(range(num_edges))
    # Union-Find or simple assignment
    # Since we just need the count and a valid assignment, we can use a dict or list.
    # Each merge (e1, e2) means color[e2] = color[e1].
    # We need to be careful with transitive merges.
    parent = list(range(num_edges))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px = find(x)
        py = find(y)
        if px != py:
            parent[px] = py
    
    for e1, e2 in merges:
        union(e1, e2)
        
    # Assign colors based on roots
    root_to_color = {}
    next_color = 0
    final_colors = [0] * num_edges
    for i in range(num_edges):
        r = find(i)
        if r not in root_to_color:
            root_to_color[r] = next_color
            next_color += 1
        final_colors[i] = root_to_color[r]
        
    return final_colors
