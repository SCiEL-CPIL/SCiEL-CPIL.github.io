"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    n = 20
    num_colors = 20
    
    # Precompute all edges
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    m = len(edge_list) # 190 edges

    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices. There are 3 ways to form a cycle from 4 vertices.
    cycles = []
    cycle_indices = [] # List of list of edge indices
    edge_to_cycles = defaultdict(list) # edge_idx -> list of cycle indices
    
    vertices = range(n)
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # The 3 distinct cycles (edge sets) on these 4 vertices
        # 1. a-b-c-d-a
        e1 = [(a,b), (b,c), (c,d), (d,a)]
        # 2. a-b-d-c-a
        e2 = [(a,b), (b,d), (d,c), (c,a)]
        # 3. a-c-b-d-a
        e3 = [(a,c), (c,b), (b,d), (d,a)]
        
        for edges in [e1, e2, e3]:
            # Normalize and get indices
            cyc_idx_set = []
            for e in edges:
                u, v = e
                if u > v:
                    u, v = v, u
                cyc_idx_set.append(edge_to_idx[(u, v)])
            
            # Check for duplicates in the cycle definition? 
            # In a simple graph, the 3 cycles for a quad are distinct edge sets.
            # However, we should ensure we don't add duplicate cycles to the global list if they appear from different quads?
            # No, each cycle is uniquely identified by its 4 edges. 
            # The 3 cycles for a specific quad are distinct. 
            # Different quads produce different sets of 4 edges.
            # So we can just append.
            
            cyc_idx = tuple(sorted(cyc_idx_set))
            cycles.append(cyc_idx)
            cycle_indices.append(cyc_idx)
            
            for e_idx in cyc_idx:
                edge_to_cycles[e_idx].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    # State variables
    # present[i] = 1 if edge i is present, 0 otherwise
    # color[i] = color of edge i
    present = [0] * m
    color = [0] * m
    
    # Initial state: Random sparse graph to allow easy start
    # Let's start with ~30% edges present and random colors
    for i in range(m):
        if random.random() < 0.3:
            present[i] = 1
            color[i] = random.randint(0, num_colors - 1)

    # Data structures for fast violation tracking
    # violation_count[cyc_idx] = number of "bad" conditions? 
    # Actually, a cycle is violated if:
    # 1. All 4 edges are present.
    # 2. The 4 colors are NOT all distinct.
    
    # We can track:
    # present_count[cyc_idx] = number of present edges in the cycle (0-4)
    # color_counts[cyc_idx][c] = number of edges in the cycle with color c (only valid if present)
    # A cycle is violated if present_count == 4 and max(color_counts) > 1 (i.e., not all distinct).
    # Since there are 4 edges, "not all distinct" means at least one color appears 2 or more times.
    
    present_count = [0] * num_cycles
    # For color counts, we can use a list of dicts or a flat array.
    # Given num_cycles ~ 14535 and colors 20, a list of lists of size 20 is 14535*20 ~ 290k ints. This is fine.
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    
    # Initialize counts
    for i in range(m):
        if present[i]:
            c = color[i]
            for cyc_idx in edge_to_cycles[i]:
                present_count[cyc_idx] += 1
                color_counts[cyc_idx][c] += 1

    # Calculate initial violations
    def is_violated(cyc_idx):
        if present_count[cyc_idx] < 4:
            return False
        # Check if all 4 edges have distinct colors
        # Sum of squares of counts? If distinct, counts are 1,1,1,1 -> sum sq = 4.
        # If not, e.g., 2,1,1,0 -> sum sq = 6.
        # So violation if sum(c^2) > 4.
        counts = color_counts[cyc_idx]
        sum_sq = 0
        for cnt in counts:
            if cnt > 0:
                sum_sq += cnt * cnt
        return sum_sq > 4

    # Maintain a set of violated cycles
    violated_cycles = set()
    for i in range(num_cycles):
        if is_violated(i):
            violated_cycles.add(i)
            
    # Objective: Maximize edges - Penalty * Violations
    # We will use a repair-based search.
    # 1. While there are violations, pick one and try to fix it.
    # 2. If no violations, try to add an edge.
    
    def get_score():
        edges = sum(present)
        return edges - 100 * len(violated_cycles)

    best_score = get_score()
    best_present = present[:]
    best_color = color[:]
    
    start_time = time.time()
    time_limit = 105.0 # Leave buffer
    
    # WalkSAT parameters
    walk_prob = 0.2 # Probability of a random move (walk) vs greedy move
    
    while time.time() - start_time < time_limit:
        # If there are violations, try to fix them
        if violated_cycles:
            # Pick a random violated cycle
            cyc_idx = random.choice(list(violated_cycles))
            
            # Get the 4 edges in this cycle
            edges_in_cyc = cycle_indices[cyc_idx]
            
            # Strategy to fix:
            # Option A: Remove one of the edges. (Reduces edge count, but removes violation)
            # Option B: Change the color of one of the edges to make them distinct.
            
            # We prefer changing color if possible, as it preserves edge count.
            # Can we change a color to fix it?
            # Current colors are not distinct.
            # We need to assign a new color to one edge such that all 4 are distinct.
            # Let the current colors be c0, c1, c2, c3.
            # We pick an edge e_i with color c_i.
            # We need a color c_new such that c_new is not in {c_j | j != i} and c_new is not c_i (obviously).
            # Actually, we just need the set of 4 colors to be distinct.
            # If we change c_i to c_new, the new set is {c_new, c_j, c_k, c_l}.
            # This set is distinct if c_new is not in {c_j, c_k, c_l}.
            
            # Let's try to find a valid color change.
            fixed = False
            # Try each edge in the cycle
            for e_idx in edges_in_cyc:
                # Get current colors of the other 3 edges
                other_colors = []
                for other_e in edges_in_cyc:
                    if other_e != e_idx:
                        other_colors.append(color[other_e])
                
                # Find a color not in other_colors
                # There are 20 colors. other_colors has 3 elements (with potential duplicates).
                # So there are at least 17 available colors.
                # We just need to pick one.
                # To be greedy, we might pick one that minimizes new violations elsewhere?
                # For now, just pick a random valid one.
                available = [c for c in range(num_colors) if c not in other_colors]
                if available:
                    old_color = color[e_idx]
                    new_color = random.choice(available)
                    if new_color == old_color:
                        # This shouldn't happen if old_color was in other_colors? 
                        # Wait, if old_color was NOT in other_colors, then the cycle might have been fixed already?
                        # No, if old_color not in other_colors, then the 4 colors were {old_color, o1, o2, o3}.
                        # If they were not distinct, then old_color must have been in {o1, o2, o3} OR one of o's was duplicated.
                        # If old_color not in others, and others have a duplicate, changing old_color won't fix the duplicate in others.
                        # So we must ensure the *resulting* set is distinct.
                        # The condition for distinctness of {c_new, o1, o2, o3} is:
                        # 1. c_new not in {o1, o2, o3}
                        # 2. o1, o2, o3 are distinct from each other.
                        
                        # Let's check condition 2 first.
                        if len(set(other_colors)) < 3:
                            continue # Cannot fix by changing this edge alone if others are already duplicated
                        
                        # If others are distinct, we just need c_new not in others.
                        # My `available` list ensures c_new not in others.
                        # So this move will fix the cycle.
                        
                        # Apply move
                        color[e_idx] = new_color
                        fixed = True
                        
                        # Update data structures
                        # 1. Update color_counts for all cycles containing e_idx
                        for c_idx in edge_to_cycles[e_idx]:
                            color_counts[c_idx][old_color] -= 1
                            color_counts[c_idx][new_color] += 1
                            
                        # 2. Update violated_cycles status for all cycles containing e_idx
                        # We need to re-check is_violated for these cycles
                        # But only if their present_count is 4.
                        for c_idx in edge_to_cycles[e_idx]:
                            if c_idx in violated_cycles:
                                if not is_violated(c_idx):
                                    violated_cycles.remove(c_idx)
                            else:
                                if is_violated(c_idx):
                                    violated_cycles.add(c_idx)
                        
                        break # Move applied, exit loop over edges
                        
            if not fixed:
                # No color change could fix it (e.g., other edges had duplicate colors)
                # Then we must remove an edge.
                e_idx = random.choice(edges_in_cyc)
                # Remove edge
                present[e_idx] = 0
                old_color = color[e_idx]
                
                # Update data structures
                for c_idx in edge_to_cycles[e_idx]:
                    present_count[c_idx] -= 1
                    color_counts[c_idx][old_color] -= 1
                    
                    # Update violation status
                    if c_idx in violated_cycles:
                        if not is_violated(c_idx):
                            violated_cycles.remove(c_idx)
                    else:
                        if is_violated(c_idx):
                            violated_cycles.add(c_idx)
            
        else:
            # No violations. Try to add an edge.
            # Pick a random edge not present.
            absent_edges = [i for i in range(m) if not present[i]]
            if not absent_edges:
                break # Complete graph achieved (unlikely)
                
            e_idx = random.choice(absent_edges)
            
            # Check if adding this edge creates any violations.
            # It only affects cycles containing e_idx.
            # We need to check if adding it makes any of those cycles violated.
            # A cycle becomes violated if:
            # 1. It was not violated before (present_count < 4 or colors distinct).
            # 2. After adding edge, present_count becomes 4.
            # 3. And colors are NOT distinct.
            
            # Let's simulate adding the edge.
            # We need to choose a color for it too.
            # To minimize risk, we should choose a color that doesn't cause violations.
            
            # Let's try all colors and see which one creates the fewest violations.
            best_c = -1
            min_new_violations = float('inf')
            
            # Pre-compute the state of cycles containing e_idx
            affected_cycles = edge_to_cycles[e_idx]
            
            for c in range(num_colors):
                # Check if adding edge with color c causes violations
                new_violations = 0
                for c_idx in affected_cycles:
                    # Current state: present_count[c_idx] is 3 (since e_idx is absent, and if it were present, it would be 4? No, e_idx is absent now).
                    # If present_count[c_idx] == 3, adding e_idx makes it 4.
                    # We need to check if colors are distinct.
                    if present_count[c_idx] == 3:
                        # Check colors of the other 3 edges
                        # color_counts[c_idx] currently reflects the 3 present edges.
                        # We add color c.
                        # The 4 colors are the 3 existing + c.
                        # Are they distinct?
                        # This is true if c is not in the existing colors AND the existing 3 are distinct.
                        
                        # Check if existing 3 are distinct
                        existing_distinct = True
                        max_count = 0
                        for cnt in color_counts[c_idx]:
                            if cnt > 1:
                                existing_distinct = False
                                break
                            if cnt > max_count:
                                max_count = cnt
                        # If existing 3 are not distinct, then adding any edge makes it violated (since 4 edges, if 3 are not distinct, 4 cannot be all distinct).
                        if not existing_distinct:
                            new_violations += 1
                            continue
                            
                        # Existing 3 are distinct.
                        # Check if c is one of them.
                        if color_counts[c_idx][c] > 0:
                            # c is already used, so duplicate.
                            new_violations += 1
                        # Else, c is new, so all 4 distinct. No violation.
                
                if new_violations < min_new_violations:
                    min_new_violations = new_violations
                    best_c = c
                    if min_new_violations == 0:
                        break # Found a perfect color
            
            # Apply the best color
            color[e_idx] = best_c
            present[e_idx] = 1
            
            # Update data structures
            for c_idx in affected_cycles:
                present_count[c_idx] += 1
                color_counts[c_idx][best_c] += 1
                
                if c_idx in violated_cycles:
                    if not is_violated(c_idx):
                        violated_cycles.remove(c_idx)
                else:
                    if is_violated(c_idx):
                        violated_cycles.add(c_idx)
                        
            # If this move created violations, we might want to revert if it's "too bad"?
            # But we chose the color with min violations.
            # If min_new_violations > 0, we have introduced violations.
            # In a strict Max-SAT, we might accept this if it allows more edges later?
            # But usually, we want to stay in the valid region if possible.
            # If we can't add any edge without creating violations, we are stuck in a local optimum?
            # Or we can try to change colors of existing edges to accommodate the new edge?
            # That's complex.
            # For now, if we create violations, the main loop will try to fix them in the next iteration.
            
        # Update best
        current_score = get_score()
        if current_score > best_score:
            best_score = current_score
            best_present = present[:]
            best_color = color[:]

    # Extract result
    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result

# EVOLVE-BLOCK-END
