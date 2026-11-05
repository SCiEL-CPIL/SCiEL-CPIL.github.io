import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, safety margin
    
    # Precompute 4-cycles
    # There are C(16,4) * 3 = 5460 4-cycles.
    # Each cycle is a tuple of 4 edge indices.
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        cycles.append((get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)))
        # Cycle 2: a-b-d-c-a
        cycles.append((get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)))
        # Cycle 3: a-c-b-d-a
        cycles.append((get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)))
        
    # Map edge index to list of cycle indices containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(idx)

    # Algorithm:
    # We want to maximize the number of distinct colors used.
    # Constraint: No 4-cycle has 4 distinct colors.
    # This is equivalent to: In every 4-cycle, at least two edges share a color.
    
    # Strategy:
    # Greedy assignment with backtracking or randomized restarts?
    # The problem size is small (120 edges).
    # A simple greedy approach that assigns the smallest possible new color might not maximize distinct colors globally.
    # However, the "last edge in cycle" logic suggests that if we can assign a new color without forcing a future conflict, we should.
    # But "future conflict" is hard to predict.
    
    # Alternative interpretation:
    # If we assign colors such that the graph of "same color" edges covers all 4-cycles (i.e., every 4-cycle has at least one pair of edges with the same color), then we are valid.
    # To maximize distinct colors, we want the color classes to be as small as possible, ideally size 1, but some must be larger to cover the 4-cycles.
    # Actually, if a color is used on k edges, it "covers" all 4-cycles that contain at least two of those edges.
    # We want to partition the 120 edges into as many color classes as possible, such that every 4-cycle is "covered" by at least one class (i.e., contains at least 2 edges of that class).
    
    # This is a set cover / packing problem.
    # Let's try a randomized greedy approach:
    # 1. Randomly permute edges.
    # 2. For each edge, try to assign a new color.
    # 3. Check if assigning a new color would leave any 4-cycle "uncovered" in a way that cannot be fixed later?
    #    No, that's complex.
    
    # Simpler heuristic:
    # Start with all edges having unique colors (120 colors). This is INVALID because many 4-cycles will have 4 distinct colors.
    # We need to merge colors until all 4-cycles are valid.
    # Merging two edges of the same color reduces the number of distinct colors by 1.
    # We want to minimize the number of merges.
    
    # Let's try a local search / hill climbing approach:
    # Start with a valid coloring (e.g., all edges color 0).
    # Try to change colors to increase distinctness.
    # Or start with a "maximally distinct" coloring and repair it.
    
    # Let's try the "repair" approach:
    # 1. Assign random colors.
    # 2. Find all invalid 4-cycles (those with 4 distinct colors).
    # 3. Pick one invalid cycle.
    # 4. Change the color of one of its edges to match another edge in the cycle.
    # 5. Repeat until no invalid cycles.
    # 6. Then try to "un-merge" colors to increase distinctness?
    #    If we change an edge's color to a NEW color, we might create new invalid cycles.
    #    So we should only change to a new color if it doesn't create new invalid cycles.
    
    # Let's implement a randomized local search:
    # State: colors array.
    # Objective: Maximize distinct colors.
    # Constraint: No 4-cycle has 4 distinct colors.
    
    # Initial state: All edges color 0. (Distinct = 1).
    # Move: Pick an edge, assign it a random color (existing or new).
    # Accept if valid and distinct count increases (or same with some probability?).
    # Since we want max distinct, we only accept if valid and distinct count increases.
    # If stuck, try to find a sequence of moves? Or just restart?
    
    # Given the small size, we can afford many restarts.
    
    def is_valid(colors):
        for e1, e2, e3, e4 in cycles:
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True

    def count_distinct(colors):
        return len(set(colors))

    best_distinct = 0
    best_colors = None
    
    # Time budget for search
    max_restarts = 100
    moves_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with all 0s
        colors = [0] * NUM_EDGES
        current_distinct = 1
        
        # Randomly shuffle edges to break symmetry
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # Try to assign new colors greedily in random order?
        # Let's try a different init: Random colors from a small palette, then expand.
        # Or just start with 0s and try to change.
        
        for _ in range(moves_per_restart):
            if time.time() > deadline:
                break
                
            # Pick a random edge
            e = random.choice(edge_order)
            old_color = colors[e]
            
            # Try assigning a new color
            # What new color? 
            # If we assign a color that is already used, distinct count doesn't change (might decrease if it was the only one).
            # If we assign a brand new color, distinct count increases by 1 (if old color is still used elsewhere).
            
            # To increase distinct count, we should try to assign a color that is currently unique to this edge?
            # No, if we assign a NEW color, and the old color is used elsewhere, distinct count increases by 1.
            # If the old color is ONLY on this edge, distinct count stays same (old removed, new added).
            
            # So, strategy:
            # 1. Find edges that have a color used by exactly 1 edge (unique colors).
            # 2. For such an edge, try to change it to a NEW color (distinct count stays same).
            # 3. For such an edge, try to change it to an EXISTING color (distinct count decreases by 1).
            # 4. For an edge with a non-unique color, try to change it to a NEW color (distinct count increases by 1).
            
            # We want to increase distinct count. So we prioritize case 4.
            
            # Check if old color is unique
            if colors.count(old_color) == 1:
                # Changing to new color keeps distinct count same.
                # Changing to existing color decreases distinct count.
                # So we should probably not touch unique colors unless we have to?
                # Or we can try to change to a new color to "prepare" for a merge?
                # Let's just skip unique colors for now to avoid decreasing distinct count.
                continue
                
            # Old color is used by >= 2 edges.
            # Try to assign a NEW color.
            # Pick a random new color index (just use a large number or track next available).
            # We can just use the current distinct count as the new color id, but we need to be consistent.
            # Let's maintain a set of used colors.
            
            # Actually, simpler: just try to set colors[e] to a value not in colors?
            # No, that's expensive.
            
            # Let's try: colors[e] = random.randint(0, 200)
            # If it's new, distinct might increase.
            
            new_color = random.randint(0, 200)
            if new_color == old_color:
                continue
                
            colors[e] = new_color
            
            if is_valid(colors):
                new_distinct = count_distinct(colors)
                if new_distinct > current_distinct:
                    current_distinct = new_distinct
                    if current_distinct > best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors[:]
                        # print(f"Restart {restart}, Move: {current_distinct}")
                else:
                    # Revert
                    colors[e] = old_color
            else:
                # Revert
                colors[e] = old_color
                
    if best_colors is None:
        # Fallback: all 0s
        best_colors = [0] * NUM_EDGES
        
    return best_colors
