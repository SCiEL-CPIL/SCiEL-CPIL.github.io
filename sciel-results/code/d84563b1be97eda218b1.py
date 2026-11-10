import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _build_edge_index():
    idx = [[-1] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx

EDGE_IDX = _build_edge_index()

# Precompute for each edge, the other 3 edges in every 4-cycle containing it
# cycle_partners[e] is a list of (e1, e2, e3)
def _build_cycle_partners():
    partners = [[] for _ in range(NUM_EDGES)]
    for i in range(N):
        for j in range(i + 1, N):
            ej = EDGE_IDX[i][j]
            for x in range(N):
                if x == i or x == j:
                    continue
                for y in range(N):
                    if y == i or y == j or y == x:
                        continue
                    # Cycle i-j-x-y-i
                    # Edges: (i,j)=ej, (j,x), (x,y), (y,i)
                    e_jx = EDGE_IDX[j][x]
                    e_xy = EDGE_IDX[x][y]
                    e_yi = EDGE_IDX[y][i]
                    # Only add if x < y to avoid duplicates? 
                    # The cycle i-j-x-y-i is the same as i-y-x-j-i?
                    # The set of edges is {(i,j), (j,x), (x,y), (y,i)}.
                    # We just need the set of 3 other edges.
                    partners[ej].append((e_jx, e_xy, e_yi))
    return partners

CYCLE_PARTNERS = _build_cycle_partners()

def _check_validity(colors, edge_idx, new_color):
    """
    Check if assigning new_color to edge_idx is valid.
    Returns True if valid.
    """
    for e1, e2, e3 in CYCLE_PARTNERS[edge_idx]:
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        # If c1, c2, c3 are distinct, then new_color must be one of them
        # to avoid a rainbow cycle {new_color, c1, c2, c3}
        if c1 != c2 and c1 != c3 and c2 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initialize with all 0s (valid)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # We want to maximize distinct colors.
    # Strategy: Local Search.
    # Try to change edges to new colors.
    
    # To speed up, we can pre-compute the list of edges to try.
    # We'll use a random walk.
    
    # Keep track of used colors to bias towards new ones
    used_colors = set([0])
    num_colors = 1
    
    # Optimization: 
    # Instead of checking all 120 colors, we can just check a few candidates.
    # If we want to add a NEW color, we must pick a color not in used_colors.
    # If we just want to move around, we can pick any valid color.
    
    # Let's try to incrementally add colors.
    # Current best: 1 color.
    # We try to find an edge that can take a color not in used_colors.
    
    # However, local search is better: 
    # Randomly pick an edge, try a random valid color.
    # If the color is new, great. If not, it might help rearrange.
    
    # To be fast, we vectorize the validity check?
    # The check is: for all (e1,e2,e3) in partners, NOT (distinct(c1,c2,c3) AND new_color not in {c1,c2,c3})
    
    # Let's just use the Python loop for validity, it's fast enough for 120 edges * 91 cycles.
    # 10k checks per step. 100k steps = 1e9 checks. Might be too slow in pure Python.
    # Let's optimize the inner loop.
    
    # Pre-compute for each edge e, the list of "triples" (e1, e2, e3).
    # We can store colors in a list for faster access than numpy array in tight loop?
    # Actually, numpy array access is slower than list in Python for single elements?
    # Let's use a list for colors during the search, convert to np at the end if needed (not needed, return list).
    
    colors_list = [0] * NUM_EDGES
    used_colors = {0}
    
    # Helper to get valid colors
    def get_valid_colors(edge_idx, colors):
        valid = []
        # We care about colors 0..119
        # But we can just test candidates.
        return valid # placeholder

    # Let's do a multi-start local search
    # Start with all 0s.
    # Loop:
    #   Pick random edge e
    #   Pick random candidate color c from 0..119
    #   If valid, update.
    #   If c is new, record.
    
    # To improve: Bias towards unused colors.
    # If we pick a color from used_colors, we are just shuffling.
    # If we pick from unused, we might expand.
    
    # Let's try to specifically target expansion.
    # If we can't find a valid unused color for any edge, we are stuck at current max.
    # Then we do random valid moves to escape local optimum.
    
    best_colors = colors_list[:]
    best_num_colors = 1
    
    for step in range(100000):
        if time.time() > deadline:
            break
        
        # 1. Try to expand: Pick a random edge, try a random unused color
        # If we have many unused colors, picking one is easy.
        # If used_colors is small, unused is large.
        
        # Strategy:
        # 50% chance: try to add a new color
        # 50% chance: random valid move (shuffle)
        
        if random.random() < 0.5 and len(used_colors) < MAX_COLOR:
            # Try to add new color
            e = random.randint(0, NUM_EDGES - 1)
            # Pick a random color not in used_colors
            # To avoid generating a list of unused colors every time, 
            # we can just pick random c in 0..119 and check if in used_colors.
            # If in used_colors, skip or retry?
            # Let's just pick c = random.randint(0, 119)
            # If c in used_colors, this move is a shuffle, not expansion.
            # We want expansion.
            
            # Efficient way: 
            # If used_colors is small, just pick random.
            # If used_colors is large, picking unused is harder.
            
            # Let's just pick random c.
            c = random.randint(0, MAX_COLOR - 1)
            
            if _check_validity(colors_list, e, c):
                old_c = colors_list[e]
                colors_list[e] = c
                if c not in used_colors:
                    used_colors.add(c)
                    if len(used_colors) > best_num_colors:
                        best_num_colors = len(used_colors)
                        best_colors = colors_list[:]
                # If c == old_c, no change.
                # If c in used_colors, it's a shuffle.
        else:
            # Random valid move
            e = random.randint(0, NUM_EDGES - 1)
            c = random.randint(0, MAX_COLOR - 1)
            if _check_validity(colors_list, e, c):
                old_c = colors_list[e]
                if c != old_c:
                    colors_list[e] = c
                    # used_colors set doesn't change size, but content might if old_c was unique?
                    # We don't track uniqueness count dynamically for speed, 
                    # but we can update best if we want.
                    # Actually, shuffling doesn't increase count.
                    # But it might allow future expansions.
                    
    # Final polish: try to fill remaining colors?
    # If we have time, try to systematically add colors.
    # For each unused color c, try to find an edge that can take it.
    
    if time.time() < deadline - 5:
        # Try to add any missing colors
        all_colors = set(range(MAX_COLOR))
        missing = list(all_colors - used_colors)
        random.shuffle(missing)
        
        for c in missing:
            if time.time() > deadline - 2:
                break
            found = False
            # Try a few random edges
            for _ in range(10):
                e = random.randint(0, NUM_EDGES - 1)
                if _check_validity(colors_list, e, c):
                    colors_list[e] = c
                    used_colors.add(c)
                    found = True
                    break
            if found:
                if len(used_colors) > best_num_colors:
                    best_num_colors = len(used_colors)
                    best_colors = colors_list[:]
    
    return best_colors
