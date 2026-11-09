import random
import time

N = 16
EDGES = []
for i in range(N):
    for j in range(i+1, N):
        EDGES.append((i, j))

# Precompute cycles for each edge
# An edge (u, v) is in a cycle if there are x, y such that u-x-v-y-u or similar.
# We determined that for edge (u,v), we check pairs (x,y) from V\{u,v}.
# The two relevant cycles are:
# 1. (u,x), (x,y), (y,v), (v,u)  -> Other edges: (u,x), (x,y), (y,v)
# 2. (u,y), (y,x), (x,v), (v,u)  -> Other edges: (u,y), (y,x), (x,v)

# Map edge to index for fast lookup
edge_index = {}
for idx, (u, v) in enumerate(EDGES):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx

# For each edge index, store the list of "other edge triplets" that constrain it.
# constraint[edge_idx] = list of tuples (idx1, idx2, idx3)
constraints = [[] for _ in range(len(EDGES))]

for idx, (u, v) in enumerate(EDGES):
    others = [k for k in range(N) if k != u and k != v]
    for x_idx in range(len(others)):
        for y_idx in range(x_idx + 1, len(others)):
            x = others[x_idx]
            y = others[y_idx]
            # Cycle 1: u-x-y-v-u
            # Edges: (u,x), (x,y), (y,v), (v,u)
            # Current edge is (v,u) which is idx.
            # Other edges: (u,x), (x,y), (y,v)
            e1 = edge_index[(u, x)]
            e2 = edge_index[(x, y)]
            e3 = edge_index[(y, v)]
            constraints[idx].append((e1, e2, e3))
            
            # Cycle 2: u-y-x-v-u
            # Edges: (u,y), (y,x), (x,v), (v,u)
            # Current edge is (v,u) which is idx.
            # Other edges: (u,y), (y,x), (x,v)
            e4 = edge_index[(u, y)]
            e5 = edge_index[(y, x)]
            e6 = edge_index[(x, v)]
            constraints[idx].append((e4, e5, e6))

def can_take_color(colors, edge_idx, new_color):
    """
    Check if edge edge_idx can be assigned new_color.
    This is only needed if we are introducing a NEW color not present in the 'other' edges?
    Actually, the condition for validity of a 4-cycle with colors c1, c2, c3, c4 is that they are not all distinct.
    If we change edge e to new_color:
    For each constraint triplet (a, b, c) in constraints[edge_idx]:
       Let colors be ca, cb, cc.
       The cycle colors will be {new_color, ca, cb, cc}.
       This is invalid if len({new_color, ca, cb, cc}) == 4.
       This happens if:
       1. ca, cb, cc are all distinct.
       2. AND new_color is not in {ca, cb, cc}.
       
       If we are using a color that is ALREADY USED elsewhere, it might be in {ca, cb, cc}.
       If we are using a NEW color (global), it is NOT in {ca, cb, cc} (assuming ca,cb,cc are from old palette? No, ca,cb,cc are current colors of other edges).
       
       So, if we are assigning a color `c`:
       For each triplet:
          if len({colors[a], colors[b], colors[c]}) == 3:
             if c not in {colors[a], colors[b], colors[c]}:
                return False
          # else (triplet has duplicates), the set size will be <= 3 even if c is new.
       
       So the check is:
       For all triplets (a,b,c):
          if colors[a], colors[b], colors[c] are distinct:
             if new_color not in (colors[a], colors[b], colors[c]):
                return False
    """
    for a, b, c in constraints[edge_idx]:
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        if ca != cb and cb != cc and ca != cc:
            # They are distinct
            if new_color != ca and new_color != cb and new_color != cc:
                return False
    return True

def construct():
    best_count = 0
    best_colors = [0] * len(EDGES)
    
    # Time limit 60s, but we should be fast.
    # Let's run 100 random restarts of the greedy algorithm.
    
    for _ in range(100):
        colors = [0] * len(EDGES)
        next_color = 1
        edge_order = list(range(len(EDGES)))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            # Try to assign next_color
            if can_take_color(colors, e_idx, next_color):
                colors[e_idx] = next_color
                next_color += 1
        
        count = len(set(colors))
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    # Optional: Local Search on best_colors to try to add more
    # Try to assign new colors to edges currently 0 (or any color)
    # We can try to bump up colors.
    
    # Let's do a few local search steps on the best solution
    # Try to change an edge to a new color
    current_colors = best_colors[:]
    used_colors = set(current_colors)
    max_color_id = max(used_colors) if used_colors else 0
    
    for step in range(1000):
        improved = False
        e_idx = random.randint(0, len(EDGES)-1)
        # Try a new color
        new_c = max_color_id + 1
        if can_take_color(current_colors, e_idx, new_c):
            current_colors[e_idx] = new_c
            used_colors.add(new_c)
            max_color_id = new_c
            improved = True
        # If not, try to swap? Or just skip.
        # To be thorough, we can try to recolor an edge to an *existing* color to free up a "distinctness" slot?
        # No, that doesn't help introduce new colors.
        
    final_count = len(set(current_colors))
    if final_count > best_count:
        best_colors = current_colors
        best_count = final_count

    return best_colors

# EVOLVE-BLOCK-END
