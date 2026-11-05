import random
import time

N = 16
E = N * (N - 1) // 2

# Precompute all 4-cycles
# A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific cyclic ordering.
# There are 3 distinct 4-cycles for any 4 vertices:
# (a,b,c,d), (a,b,d,c), (a,c,b,d)
# Edges in (a,b,c,d) are (a,b), (b,c), (c,d), (d,a)

# Map edge (i,j) to index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Wait, simpler index:
# i < j
# idx = 0
# for i in range(N):
#   for j in range(i+1, N):
#     ...
# Let's just build a dictionary.
edge_map = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_map[(i, j)] = idx
        idx += 1

# List of all 4-cycles as lists of 4 edge indices
cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = edge_map[(a, b)]
                e2 = edge_map[(b, c)]
                e3 = edge_map[(c, d)]
                e4 = edge_map[(d, a)]
                cycles.append([e1, e2, e3, e4])
                
                # Cycle 2: a-b-d-c-a
                e1 = edge_map[(a, b)]
                e2 = edge_map[(b, d)]
                e3 = edge_map[(d, c)] # c<d
                e4 = edge_map[(c, a)] # a<c
                cycles.append([e1, e2, e3, e4])
                
                # Cycle 3: a-c-b-d-a
                e1 = edge_map[(a, c)]
                e2 = edge_map[(c, b)] # b<c
                e3 = edge_map[(b, d)]
                e4 = edge_map[(d, a)]
                cycles.append([e1, e2, e3, e4])

# Reverse map: for each edge, list of cycles containing it
edge_to_cycles = [[] for _ in range(E)]
for k, cyc in enumerate(cycles):
    for e in cyc:
        edge_to_cycles[e].append(k)

def is_valid(colors, e, new_color):
    # Check if changing edge e to new_color violates any cycle
    for k in edge_to_cycles[e]:
        cyc = cycles[k]
        # Get colors of the other 3 edges
        # And the new color for e
        c_other = []
        for x in cyc:
            if x == e:
                c_other.append(new_color)
            else:
                c_other.append(colors[x])
        if len(set(c_other)) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Start with all 0
    best_colors = [0] * E
    best_count = 1
    
    # Try to improve
    # We will iterate through edges and try to assign new colors
    # To maximize colors, we maintain a set of used colors
    
    for restart in range(20): # Limit restarts
        if time.time() > deadline:
            break
            
        colors = [0] * E
        used_colors = {0}
        
        # Random order of edges
        edge_order = list(range(E))
        random.shuffle(edge_order)
        
        # We try to assign a new unique color to each edge if possible
        # If not, we leave it or try to fix later?
        # Simple greedy: for each edge in random order, try to assign a new color.
        # If it fails, try existing colors? No, we want to maximize count.
        # If it fails, leave it as is (0).
        
        current_count = 1
        next_color = 1
        
        for e in edge_order:
            if time.time() > deadline:
                break
            
            # Try to assign a new color
            # We can try a few random new colors from the pool 0..119
            # But we want distinct ones.
            # Let's try the next available color first, then random others.
            
            assigned = False
            # Try next_color
            if next_color < 120 and is_valid(colors, e, next_color):
                colors[e] = next_color
                used_colors.add(next_color)
                current_count += 1
                next_color += 1
                assigned = True
            
            if not assigned:
                # Try a few random unused colors
                # Pick random 10 unused colors
                unused = [c for c in range(120) if c not in used_colors]
                random.shuffle(unused)
                for c in unused[:10]:
                    if is_valid(colors, e, c):
                        colors[e] = c
                        used_colors.add(c)
                        current_count += 1
                        assigned = True
                        break
            
            # If still not assigned, leave as 0 (or whatever it was)
            
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
            
    return best_colors
