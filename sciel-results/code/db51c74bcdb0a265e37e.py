import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    # Precompute edges
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges) # 120
    
    # Precompute K4s
    # k4s_all: list of lists of edge indices
    # edge_k4s: list of lists of k4 indices for each edge
    k4s_all = []
    edge_k4s = [[] for _ in range(num_edges)]
    
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
        k4_idx = len(k4s_all)
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    # Initial colors: (i + j) % 17
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17

    # We need to track for each K4 the count of each color to quickly check validity.
    # A K4 is valid if the number of distinct colors >= 5.
    # If we change an edge color in a K4, we can update the counts.
    # However, maintaining counts for all 1820 K4s * 17 colors is ~30k ints.
    # Updating is O(1) per K4.
    # Checking validity is O(1) if we track distinct count.
    
    # Let's maintain:
    # k4_color_counts[k4_idx][color] = count
    # k4_distinct[k4_idx] = number of colors with count > 0
    
    max_color_val = 17
    # Use a list of lists of ints.
    # To save memory and time, maybe just use a dict or small list.
    # Since colors are 0..16 initially, and we might change them, let's allow up to 20 colors.
    # If we use random integers, they could be large.
    # But we want to minimize distinct colors.
    # Let's restrict the color space to 0..19 during search to keep counts small.
    # If we find a solution with < 13 colors, we can compact.
    
    # Actually, the "raw value" is the number of distinct colors used.
    # The evaluator likely counts distinct values in the returned list.
    # So we can use any integers, but smaller distinct count is better.
    
    # Let's use a fixed color palette of size P=20 for the search.
    # If we can get down to 13 distinct colors within this palette, great.
    # If we need to explore outside, it's unlikely to help minimize distinct count.
    
    P = 20
    k4_color_counts = [[0] * P for _ in range(len(k4s_all))]
    k4_distinct = [0] * len(k4s_all)
    
    def update_k4_counts(k4_idx, edge_idx, old_color, new_color):
        """
        Update counts for a single K4 when an edge color changes.
        Returns the change in distinct count (delta).
        """
        # Clamp colors to palette? No, if new_color >= P, we can't track it.
        # So we must ensure new_color < P.
        
        delta = 0
        if old_color != new_color:
            # Decrement old
            if old_color < P:
                if k4_color_counts[k4_idx][old_color] == 1:
                    k4_distinct[k4_idx] -= 1
                    delta -= 1
                k4_color_counts[k4_idx][old_color] -= 1
            
            # Increment new
            if new_color < P:
                if k4_color_counts[k4_idx][new_color] == 0:
                    k4_distinct[k4_idx] += 1
                    delta += 1
                k4_color_counts[k4_idx][new_color] += 1
        return delta

    # Initialize counts
    for k4_idx, e_indices in enumerate(k4s_all):
        for ei in e_indices:
            c = colors[ei]
            if c < P:
                k4_color_counts[k4_idx][c] += 1
        # Calculate distinct
        distinct = 0
        for c in range(P):
            if k4_color_counts[k4_idx][c] > 0:
                distinct += 1
        k4_distinct[k4_idx] = distinct

    # Check initial validity
    # All K4s should have >= 5 distinct colors.
    # (i+j)%17 is known to be valid for this task?
    # The seed says 17 colors.
    # Let's verify.
    valid = True
    for k in range(len(k4s_all)):
        if k4_distinct[k] < 5:
            valid = False
            break
    if not valid:
        # Fallback: just return the seed colors, though it might be invalid?
        # The problem states the seed is valid.
        pass

    best_colors = colors[:]
    best_distinct = len(set(colors))
    
    # Maintain global color counts to know which colors are "active" (used somewhere)
    # This helps in choosing new colors.
    global_color_counts = [0] * P
    for c in colors:
        if c < P:
            global_color_counts[c] += 1

    def is_valid_all():
        for k in range(len(k4s_all)):
            if k4_distinct[k] < 5:
                return False
        return True

    # Local Search Loop
    # We want to minimize distinct colors.
    # Strategy:
    # 1. Pick a random edge.
    # 2. Pick a random new color from the palette.
    # 3. Check if changing this edge to new_color keeps all affected K4s valid.
    #    "Valid" means distinct count >= 5.
    #    We can simulate the change on the affected K4s.
    # 4. If valid, accept.
    # 5. Periodically check if we reduced the global distinct count.
    
    # Optimization:
    # Instead of checking all 91 K4s for each move, we can just do it.
    # 91 * (constant time) is fast.
    
    # To be more aggressive, we can bias the new color selection.
    # We want to eliminate rare colors.
    
    while time.time() < deadline:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Choose new color
        # Bias towards colors that are already used to consolidate?
        # Or random?
        # If we pick a color that is not used, we increase distinct count (bad).
        # So we should only pick from currently used colors?
        # If we only pick from used colors, we might get stuck.
        # But minimizing distinct count means we shouldn't introduce new colors.
        # So restrict new_color to colors with global_color_counts[c] > 0.
        
        used_colors = [c for c in range(P) if global_color_counts[c] > 0]
        
        if not used_colors:
            break
            
        new_color = random.choice(used_colors)
        
        if new_color == old_color:
            continue
            
        # Simulate change
        # Affected K4s are in edge_k4s[e_idx]
        # We need to check if k4_distinct[k4] remains >= 5 after change.
        
        # For each affected K4, calculate the potential new distinct count.
        # old_distinct = k4_distinct[k4]
        # If old_color count in K4 becomes 0, distinct decreases by 1.
        # If new_color count in K4 was 0, distinct increases by 1.
        
        valid_move = True
        # Store changes to revert if invalid
        changes = []
        
        for k4_idx in edge_k4s[e_idx]:
            old_count = k4_color_counts[k4_idx][old_color]
            new_count = k4_color_counts[k4_idx][new_color]
            
            # Calculate new distinct
            new_distinct = k4_distinct[k4_idx]
            
            if old_count == 1:
                new_distinct -= 1
            if new_count == 0:
                new_distinct += 1
                
            if new_distinct < 5:
                valid_move = False
                break
                
            # Store for potential update
            # We don't actually update yet, just check.
            # If valid, we will apply.
            
        if valid_move:
            # Apply change
            colors[e_idx] = new_color
            
            # Update global counts
            global_color_counts[old_color] -= 1
            global_color_counts[new_color] += 1
            
            # Update K4 counts
            for k4_idx in edge_k4s[e_idx]:
                # Update counts
                if k4_color_counts[k4_idx][old_color] == 1:
                    k4_distinct[k4_idx] -= 1
                k4_color_counts[k4_idx][old_color] -= 1
                
                if k4_color_counts[k4_idx][new_color] == 0:
                    k4_distinct[k4_idx] += 1
                k4_color_counts[k4_idx][new_color] += 1
            
            # Check if we improved best
            # Count distinct colors in global_color_counts
            current_distinct = sum(1 for c in range(P) if global_color_counts[c] > 0)
            
            if current_distinct < best_distinct:
                best_distinct = current_distinct
                best_colors = colors[:]
                
                # If we hit the lower bound 13, we can stop?
                # The task says 13 <= f(16) <= 15.
                # If we find 13, it's optimal.
                if best_distinct == 13:
                    break

    # Final compression
    # Map used colors to 0..k-1
    used = set(best_colors)
    mapping = {c: i for i, c in enumerate(sorted(used))}
    final_colors = [mapping[c] for c in best_colors]
    
    return final_colors

# EVOLVE-BLOCK-END
