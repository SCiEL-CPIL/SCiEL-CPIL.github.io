import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    num_edges = 120
    
    # Precompute edge indices
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1
            
    # Precompute K4s and their edge indices
    k4s = []
    for a, b, c, d in combinations(range(n), 4):
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
        
    # Precompute for each edge, the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
            
    # Helper to check if a K4 is valid given current colors
    def is_k4_valid(k4_idx, colors):
        edge_idxs = k4s[k4_idx]
        # Count distinct colors
        # Optimization: use a small set or manual check
        c0 = colors[edge_idxs[0]]
        c1 = colors[edge_idxs[1]]
        c2 = colors[edge_idxs[2]]
        c3 = colors[edge_idxs[3]]
        c4 = colors[edge_idxs[4]]
        c5 = colors[edge_idxs[5]]
        
        distinct = 1
        if c1 != c0: distinct += 1
        if c2 != c0 and c2 != c1: distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        
        return distinct >= 5

    def count_violations(colors):
        count = 0
        for k4_idx in range(len(k4s)):
            if not is_k4_valid(k4_idx, colors):
                count += 1
        return count

    # Initialize with 16 colors
    # A simple random initialization with 16 colors
    random.seed(42)
    colors = [random.randint(0, 15) for _ in range(num_edges)]
    
    # Repair phase: fix violations by changing colors
    # We want to end up with a valid coloring.
    # If we start random, it might have many violations.
    # Let's start with a known valid one: (i+j) % 17, then try to compress.
    # Actually, (i+j) % 17 uses 17 colors. Let's use 16 colors in init.
    # Try to find a valid 16-coloring first.
    
    # Better Init: Use a structure.
    # Let's try to construct a valid 16-coloring using local search from a random 16-coloring.
    
    current_colors = colors[:]
    current_violations = count_violations(current_colors)
    
    # Repair loop
    repair_iterations = 0
    while current_violations > 0 and time.time() < deadline - 50:
        # Pick a random violating K4
        # Instead of finding all, just pick a random edge and check its K4s?
        # No, pick a random K4, if it's violating, fix one of its edges.
        
        # Find a violating K4
        # Optimization: maintain a list of violating K4s?
        # For now, just scan. 1820 K4s is small.
        violating_k4s = [i for i in range(len(k4s)) if not is_k4_valid(i, current_colors)]
        
        if not violating_k4s:
            break
            
        k4_idx = random.choice(violating_k4s)
        edge_idxs = k4s[k4_idx]
        
        # Try to change color of one edge in this K4 to fix it
        # We need to increase the number of distinct colors in this K4.
        # If it has 4 colors, we need 1 more. If 3, need 2.
        # Changing one edge can add at most 1 new color to the set.
        # So if distinct < 4, we might need multiple changes or a specific change.
        
        # Simple heuristic: try changing each edge in the K4 to a random new color
        # and check if the K4 becomes valid and no other K4 is broken.
        
        fixed = False
        for ei in edge_idxs:
            old_c = current_colors[ei]
            # Try a few new colors
            for attempt in range(10):
                new_c = random.randint(0, 15)
                if new_c == old_c:
                    continue
                current_colors[ei] = new_c
                
                # Check if this K4 is now valid
                if not is_k4_valid(k4_idx, current_colors):
                    current_colors[ei] = old_c
                    continue
                    
                # Check if this move broke any other K4 containing ei
                valid_move = True
                for other_k4_idx in edge_to_k4s[ei]:
                    if other_k4_idx == k4_idx:
                        continue
                    if not is_k4_valid(other_k4_idx, current_colors):
                        valid_move = False
                        break
                        
                if valid_move:
                    # Calculate new violation count (approximate or exact)
                    # For speed, just accept and recalculate total later?
                    # No, we need to track it.
                    # Let's just accept and update count_violations at the end of repair step?
                    # That's expensive.
                    # Let's just accept if it fixes this K4 and doesn't break others.
                    # This is a greedy repair.
                    fixed = True
                    break
                else:
                    current_colors[ei] = old_c
            
            if fixed:
                break
        
        if not fixed:
            # Stuck in repair, restart or perturb
            # Perturb: change a random edge
            ei = random.randint(0, 119)
            current_colors[ei] = random.randint(0, 15)
            
        # Update violation count (expensive but necessary for loop condition)
        # Optimization: only recalculate if we made a change?
        # We can just break out and check.
        current_violations = count_violations(current_colors)
        
    if current_violations > 0:
        # If we couldn't repair, fall back to seed solution
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    # Now we have a valid 16-coloring (or fewer if random used fewer)
    # Try to reduce colors
    current_palette = sorted(set(current_colors))
    num_colors = len(current_palette)
    best_colors = num_colors
    best_col_list = current_colors[:]
    
    # Local search to reduce colors
    while time.time() < deadline:
        if num_colors <= 13:
            break
            
        # Pick a color to eliminate
        # Prefer colors with fewer edges? Or random?
        color_counts = {}
        for c in current_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
            
        # Try to eliminate a color
        # Shuffle colors to avoid bias
        palette = current_palette[:]
        random.shuffle(palette)
        
        eliminated = False
        for target_color in palette:
            if target_color not in color_counts:
                continue
                
            edges_with_color = [i for i in range(num_edges) if current_colors[i] == target_color]
            
            # Try to remap each edge
            for ei in edges_with_color:
                if time.time() > deadline:
                    break
                    
                old_c = current_colors[ei]
                
                # Try changing to any other color in palette
                for new_c in current_palette:
                    if new_c == target_color:
                        continue
                        
                    current_colors[ei] = new_c
                    
                    # Check K4s containing ei
                    valid = True
                    for k4_idx in edge_to_k4s[ei]:
                        if not is_k4_valid(k4_idx, current_colors):
                            valid = False
                            break
                            
                    if valid:
                        # Check if target_color is still used
                        if target_color not in current_colors:
                            # Success!
                            current_palette = sorted(set(current_colors))
                            num_colors = len(current_palette)
                            if num_colors < best_colors:
                                best_colors = num_colors
                                best_col_list = current_colors[:]
                            eliminated = True
                            break
                        else:
                            # Not eliminated, revert? Or keep if it helps?
                            # Usually we want to eliminate. If not eliminated, revert to keep state simple?
                            # Actually, if we don't eliminate, we might be moving towards elimination.
                            # But for simplicity, let's only accept if it eliminates.
                            current_colors[ei] = old_c
                    else:
                        current_colors[ei] = old_c
                        
                if eliminated:
                    break
                    
            if eliminated:
                break
                
        if not eliminated:
            # Perturb to escape local optimum
            # Change a random edge to a random color
            ei = random.randint(0, 119)
            old_c = current_colors[ei]
            new_c = random.randint(0, num_colors - 1)
            current_colors[ei] = new_c
            
            # Check validity
            valid = True
            for k4_idx in edge_to_k4s[ei]:
                if not is_k4_valid(k4_idx, current_colors):
                    valid = False
                    break
                    
            if not valid:
                current_colors[ei] = old_c
            else:
                # Update palette
                current_palette = sorted(set(current_colors))
                num_colors = len(current_palette)
                if num_colors < best_colors:
                    best_colors = num_colors
                    best_col_list = current_colors[:]
    
    # Final check
    if count_violations(best_col_list) > 0:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_col_list
