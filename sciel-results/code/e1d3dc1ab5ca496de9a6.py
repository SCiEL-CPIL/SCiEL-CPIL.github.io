import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices
edge_indices = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_indices[(i, j)] = idx
        idx += 1

# Precompute cycles containing each edge
# cycles_containing_edge[e_idx] = list of tuples (e1, e2, e3)
# where (e_idx, e1, e2, e3) form a 4-cycle
cycles_containing_edge = [[] for _ in range(NUM_EDGES)]

for u in range(N):
    for v in range(u + 1, N):
        e_idx = edge_indices[(u, v)]
        for x in range(N):
            if x == u or x == v:
                continue
            for y in range(x + 1, N):
                if y == u or y == v:
                    continue
                # Cycle 1: u-v-x-y-u
                e1 = edge_indices[(v, x) if v < x else (x, v)]
                e2 = edge_indices[(x, y)]
                e3 = edge_indices[(y, u) if y < u else (u, y)]
                cycles_containing_edge[e_idx].append((e1, e2, e3))
                
                # Cycle 2: u-v-y-x-u
                e1 = edge_indices[(v, y) if v < y else (y, v)]
                e2 = edge_indices[(y, x) if y < x else (x, y)] # same as (x,y)
                e3 = edge_indices[(x, u) if x < u else (u, x)]
                cycles_containing_edge[e_idx].append((e1, e2, e3))

def is_valid_change(colors, e_idx, new_color):
    """
    Check if changing colors[e_idx] to new_color maintains validity.
    Only cycles containing e_idx need to be checked.
    """
    for e1, e2, e3 in cycles_containing_edge[e_idx]:
        c1, c2, c3 = colors[e1], colors[e2], colors[e3]
        # If new_color is not in {c1, c2, c3}, and c1,c2,c3 are distinct, 
        # then {new_color, c1, c2, c3} has 4 distinct colors.
        # Actually, if c1,c2,c3 are not all distinct, say c1==c2, then set is {new, c1, c3}.
        # Max size 3. So no rainbow.
        # If c1,c2,c3 are distinct, and new_color is not one of them, size is 4.
        if c1 != c2 and c2 != c3 and c1 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
        # If c1,c2,c3 are not distinct, e.g., c1==c2 != c3.
        # Set is {new, c1, c3}. Max size 3. Always valid.
        # So we only need to check if the other 3 are distinct and new is different.
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initialize with 0
    colors = [0] * NUM_EDGES
    color_counts = [0] * MAX_COLOR
    for c in colors:
        color_counts[c] += 1
        
    used_colors = set(c for c in range(MAX_COLOR) if color_counts[c] > 0)
    best_score = len(used_colors)
    best_colors = colors[:]
    
    all_edges = list(range(NUM_EDGES))
    random.shuffle(all_edges)
    
    while time.time() < deadline:
        # Find unused colors
        unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
        if not unused:
            break
            
        # Pick a target color to introduce
        target_color = random.choice(unused)
        
        # Pick a random starting edge
        start_edge = random.choice(all_edges)
        
        # Attempt to inject target_color
        changes = [] # List of (edge_idx, old_color)
        queue = [(start_edge, target_color)]
        visited_in_attempt = {start_edge}
        success = False
        max_depth = 30
        
        while queue and len(changes) < max_depth:
            e_idx, c = queue.pop(0)
            
            if colors[e_idx] == c:
                continue
                
            # Check validity
            if is_valid_change(colors, e_idx, c):
                # Apply change
                old_c = colors[e_idx]
                colors[e_idx] = c
                changes.append((e_idx, old_c))
                
                # Update counts temporarily? 
                # We need to update counts to check validity of subsequent changes?
                # is_valid_change uses `colors` array. So we must update `colors`.
                # We don't strictly need to update `color_counts` for validity checks,
                # only for the final score. But we need to know if we actually introduced the color.
                
                # If we successfully changed the first edge, we might consider it a success 
                # if we don't need to propagate further?
                # No, we must ensure the *entire* state is valid.
                # The changes so far are valid w.r.t each other?
                # is_valid_change only checks cycles containing the *changed* edge.
                # Since we only change one edge at a time, and check against current colors,
                # the state remains valid after each step.
                
                # We continue to propagate if we want to "spread" the color?
                # Actually, just introducing the color to one edge is enough to increase distinct count by 1.
                # Do we need to change more edges?
                # If we just change one edge, the score increases by 1.
                # Is it better to stop there?
                # Yes, simpler is better.
                
                # However, if we stop, we might have left the graph in a state 
                # where it's hard to introduce *other* colors later?
                # The greedy approach usually works well.
                
                # Let's just take the first valid change and stop.
                # This makes the attempt very fast.
                success = True
                break
                
            else:
                # Find a conflict
                # We need to pick an edge from a conflicting cycle to try to change it to c.
                conflicting_cycle = None
                for e1, e2, e3 in cycles_containing_edge[e_idx]:
                    c1, c2, c3 = colors[e1], colors[e2], colors[e3]
                    if c1 != c2 and c2 != c3 and c1 != c3:
                        if c != c1 and c != c2 and c != c3:
                            conflicting_cycle = (e1, e2, e3)
                            break
                            
                if conflicting_cycle:
                    # Pick a random edge from the conflict to recurse on
                    next_e = random.choice(conflicting_cycle)
                    if next_e not in visited_in_attempt:
                        visited_in_attempt.add(next_e)
                        queue.append((next_e, c))
                else:
                    # Should not happen if is_valid_change returned False
                    break
        
        if success:
            # Update counts
            for e_idx, old_c in changes:
                color_counts[old_c] -= 1
                if color_counts[old_c] == 0:
                    used_colors.discard(old_c)
                
                color_counts[c] += 1
                if color_counts[c] == 1:
                    used_colors.add(c)
                    
            new_score = len(used_colors)
            if new_score > best_score:
                best_score = new_score
                best_colors = colors[:]
        else:
            # Revert changes
            for e_idx, old_c in changes:
                colors[e_idx] = old_c
                # Counts don't need reverting if we didn't update them?
                # We didn't update counts during the attempt.
                # So we just revert colors.
                pass

    return best_colors
