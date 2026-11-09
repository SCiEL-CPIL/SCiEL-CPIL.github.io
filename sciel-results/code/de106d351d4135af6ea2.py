import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    # 1. Precompute Edge Indices and K4 structures
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges) # 120
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    k4_edges_list = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_index[(a, b)],
            edge_index[(a, c)],
            edge_index[(a, d)],
            edge_index[(b, c)],
            edge_index[(b, d)],
            edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges_list.append(e_indices)
    
    num_k4s = len(k4s) # 1820
    
    # Map edge index to list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edges_list):
        for ei in e_indices:
            edge_to_k4s[ei].append(k4_idx)

    # Initialize colors with the seed construction: (i + j) % 17
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # Helper: Check if a specific K4 is valid given current colors
    def is_k4_valid(k4_idx, cols):
        e_indices = k4_edges_list[k4_idx]
        c_set = set()
        for ei in e_indices:
            c_set.add(cols[ei])
            if len(c_set) >= 5:
                return True
        return len(c_set) >= 5

    # Helper: Check if changing edge `e_idx` from `old_c` to `new_c` is valid
    def is_change_valid(e_idx, old_c, new_c, cols):
        if old_c == new_c:
            return True
        cols[e_idx] = new_c
        valid = True
        for k4_idx in edge_to_k4s[e_idx]:
            if not is_k4_valid(k4_idx, cols):
                valid = False
                break
        if not valid:
            cols[e_idx] = old_c
        return valid

    # Helper: Count distinct colors
    def count_colors(cols):
        return len(set(cols))

    best_colors = colors[:]
    best_val = count_colors(best_colors)
    
    # Optimization: Track max color used to limit search space
    current_max_color = max(colors)
    
    # Main Search Loop
    # We alternate between Greedy Compression and Random Search
    # Greedy Compression tries to lower edge colors to 0, 1, ...
    # This tends to collapse the palette.
    
    step = 0
    while time.time() < deadline:
        # Phase 1: Greedy Compression
        # Try to reduce the color of each edge to the smallest possible valid value
        # This is deterministic and fast if no changes are made, but we randomize order to avoid bias
        # We do a few passes of compression
        
        if step % 10 == 0: # Every 10 iterations of the main loop, do a heavy compression
            # Shuffle edge order for this pass
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            
            for e_idx in edge_order:
                old_c = colors[e_idx]
                # Try colors 0 to old_c - 1
                for new_c in range(old_c):
                    if is_change_valid(e_idx, old_c, new_c, colors):
                        # Accepted
                        old_c = new_c
                        # Update max color if necessary (it can only decrease or stay same in this loop)
                        break
                # If old_c was reduced, we don't immediately update global max, 
                # but we can recalc periodically or just let it be an upper bound.
            
            # Recalculate max color and distinct count
            curr_max = max(colors)
            curr_val = count_colors(colors)
            if curr_val < best_val:
                best_val = curr_val
                best_colors = colors[:]
                current_max_color = curr_max
            else:
                current_max_color = curr_max # Update bound for random phase
            
        else:
            # Phase 2: Random Perturbation
            # Pick a random edge and a random color within the current range
            # This allows escaping local minima where greedy compression is stuck
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            # Choose a candidate color
            # Bias towards lower colors to encourage compression, 
            # but allow higher colors to explore? 
            # Actually, if we want to minimize distinct count, we want to use existing low colors.
            # Let's pick from [0, current_max_color]
            
            if current_max_color > 0:
                new_c = random.randint(0, current_max_color)
            else:
                new_c = 0
                
            if new_c != old_c:
                if is_change_valid(e_idx, old_c, new_c, colors):
                    # Accept
                    # Check if we improved the distinct count
                    # To save time, we only check distinct count occasionally or if max color changed?
                    # Actually, distinct count can only decrease if we stop using a color.
                    # It can increase if we introduce a new color (but we are limited to current_max).
                    # So distinct count is non-increasing in the random phase if we restrict to [0, max].
                    # Wait, if we have colors {0, 5} and change 5 to 0, distinct becomes {0}.
                    # If we have {0, 1} and change 1 to 2 (if 2 <= max), distinct becomes {0, 2}.
                    # So distinct count can increase.
                    # We should check validity of the "best" improvement.
                    
                    # Efficient check: 
                    # If new_c was not in the palette before, and old_c is now gone, distinct count might stay same or decrease.
                    # It's safer to just count periodically.
                    
                    # Let's count every 100 random steps
                    step += 1
                    if step % 100 == 0:
                        curr_val = count_colors(colors)
                        if curr_val < best_val:
                            best_val = curr_val
                            best_colors = colors[:]
                            current_max_color = max(colors)
                        else:
                            # If not improved, check if max color changed to update bound
                            # This is O(N), but N=120 is small.
                            # However, we can skip this to save time, just use a loose bound.
                            pass
                else:
                    # Rejected
                    step += 1
                    if step % 100 == 0:
                         # Still update step counter
                         pass
        
        # Safety check: if we hit 13 colors, we can stop early as it's the theoretical lower bound
        if best_val <= 13:
            break

    # Final Compression on the best found solution to ensure minimal labels
    # The evaluator counts distinct values. 
    # If we have colors [0, 2, 5], distinct is 3.
    # If we remap to [0, 1, 2], distinct is 3.
    # The value returned is the list. The "raw value" is the distinct count.
    # So remapping doesn't change the score, but it's good practice.
    
    # However, we must ensure the final returned list is valid.
    # best_colors is a snapshot of a valid state.
    
    # One final greedy pass on best_colors to ensure it's as compressed as possible?
    # The best_colors was taken at a point where it was valid.
    # We can try to compress it further without risking invalidity?
    # Yes, run the compression logic on best_colors one last time.
    
    final_colors = best_colors[:]
    # Run a quick deterministic compression on final_colors
    # We need to be careful: is_change_valid modifies the array it's passed.
    # So we pass final_colors.
    
    # We need to re-run the compression loop on final_colors
    # But we need to ensure we don't exceed time.
    # The compression loop is O(N * K4s_per_edge * Cost_Check).
    # Cost_Check is small.
    # Let's do one pass.
    
    for e_idx in range(num_edges):
        old_c = final_colors[e_idx]
        for new_c in range(old_c):
            if is_change_valid(e_idx, old_c, new_c, final_colors):
                old_c = new_c
                break
                
    # Remap colors to 0..k-1 to be safe, though distinct count is the metric
    unique_colors = sorted(list(set(final_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in final_colors]
    
    return final_colors

# EVOLVE-BLOCK-END
