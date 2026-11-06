import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    num_edges = 120
    
    # Precompute edge indices for each K4
    # Edge (i, j) with i < j has index: sum_{k=0}^{i-1} (15-k) + (j-i-1)
    # = 15*i - i*(i-1)//2 + j - i - 1
    offsets = [0] * n
    for i in range(n):
        offsets[i] = 15 * i - i * (i - 1) // 2

    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return offsets[i] + (j - i - 1)

    # Generate all K4s
    k4_list = []
    k4_edge_indices = []
    import itertools
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        e_idxs = [get_edge_idx(u, v) for u, v in edges]
        k4_list.append(e_idxs)
    
    num_k4 = len(k4_list)
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_idxs in enumerate(k4_edge_indices):
        for e_idx in e_idxs:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # We want to minimize the number of colors used.
    # The seed program uses (i+j) % 17, which uses 17 colors? No, wait.
    # The task says "Minimize the number of colours used".
    # The seed program returns a list of colors.
    # The evaluator measures the "raw value". 
    # "A valid colouring certifies f(16) <= colours used."
    # "The evaluator measures the raw value (lower is better)."
    # So the score is the number of distinct colors used in the returned list?
    # Or is the score the value of the objective function?
    # "Minimize the number of colours used." -> Score = number of distinct colors.
    # "The evaluator measures the raw value (lower is better)."
    # So if I use 14 colors, score is 14. If I use 13, score is 13.
    # The seed program uses `(i + j) % 17`. The distinct colors used are 0..16, so 17 colors.
    # But the score is 1.0.
    # "search score is seed/value".
    # If seed value is 17, and my value is 14, score = 17/14 = 1.21.
    # If my value is 13, score = 17/13 = 1.30.
    # If my value is 15, score = 17/15 = 1.13.
    # The current program in the prompt is the "seed program" with score 1.0000.
    # The seed program in the prompt is the one I am rewriting.
    # Wait, the "CURRENT PROGRAM (score 1.0000)" is the one I am given to rewrite.
    # It returns `[(i + j) % 17 ...]` as a fallback, but it tries to search.
    # If the search fails, it returns the fallback.
    # The fallback uses 17 colors.
    # So the seed value is likely 17? Or maybe the seed program found a solution with 17 colors?
    # Or maybe the seed program is the one that returns the fallback?
    # Let's assume the seed value is the number of colors in the fallback, which is 17.
    # Actually, the task says "The general bounds give 13 <= f(16) <= 15".
    # So the optimal number of colors is between 13 and 15.
    # If I find a valid coloring with 15 colors, score = 17/15 = 1.133.
    # If I find a valid coloring with 14 colors, score = 17/14 = 1.214.
    # If I find a valid coloring with 13 colors, score = 17/13 = 1.307.
    
    # So the goal is to find a valid coloring with as few colors as possible.
    # The "value" measured by the evaluator is the number of distinct colors used.
    
    # Strategy:
    # 1. Try to find a valid coloring with 13 colors.
    # 2. If not, try 14 colors.
    # 3. If not, try 15 colors.
    # 4. If not, return a valid coloring with 16 or 17 colors (fallback).
    
    # We will use a local search approach.
    # Objective: Minimize the number of violations (K4s with < 5 colors).
    # Constraint: Use at most C colors.
    # If we find a solution with 0 violations using C colors, we return it.
    # We try C = 13, 14, 15.
    
    # To speed up, we can use a bitmask for colors if C <= 64.
    # C is 13, 14, or 15. So bitmask works.
    
    # We will maintain the state of each K4: the set of colors present.
    # Actually, we just need to know if the number of distinct colors is >= 5.
    # We can maintain a count of distinct colors for each K4.
    # When we change an edge color, we update the K4s containing that edge.
    
    # Let's implement a fast local search.
    
    def count_colors(colors, e_idxs):
        mask = 0
        for e in e_idxs:
            mask |= (1 << colors[e])
        return bin(mask).count('1')
    
    def total_violations(colors):
        violations = 0
        for e_idxs in k4_edge_indices:
            d = count_colors(colors, e_idxs)
            if d < 5:
                violations += (5 - d)
        return violations
    
    # We'll use a simple hill climbing with restarts.
    # To make it faster, we'll precompute the list of K4s for each edge.
    
    best_colors = None
    best_num_colors = 100
    
    for C in [13, 14, 15]:
        # Initialize with random colors
        colors = np.random.randint(0, C, size=num_edges)
        
        # Compute initial violations
        # We can speed this up by maintaining a count of distinct colors for each K4.
        # But for simplicity, let's just compute total violations.
        # 1820 K4s * 6 edges = 10920 operations. Very fast.
        
        cur_viol = total_violations(colors)
        
        if cur_viol == 0:
            return colors.tolist()
        
        # Hill climbing
        # We'll try to reduce violations.
        # If stuck, restart.
        
        stuck_count = 0
        max_stuck = 5000
        
        while time.time() < deadline:
            # Pick a random edge
            e_idx = np.random.randint(0, num_edges)
            old_color = colors[e_idx]
            
            # Try a new color
            # We can try all C colors and pick the best one.
            # Or just pick a random one.
            # Let's try all C colors and pick the one that reduces violations the most.
            # This is a steepest descent step.
            
            best_new_color = old_color
            best_delta = 0
            
            # We need to compute the change in violations if we change edge e_idx to new_color.
            # This only affects K4s containing e_idx.
            # There are at most 15 * 14 / 2? No.
            # An edge (u, v) is in K4s formed by {u, v, x, y} where x, y are distinct from u, v.
            # Number of such K4s is C(14, 2) = 91.
            # So we only need to check 91 K4s.
            
            affected_k4s = edge_to_k4s[e_idx]
            
            # Compute current distinct colors for affected K4s
            # And what they would be with the new color.
            
            # To be efficient, we can precompute the color counts for each K4?
            # Or just recompute the distinct count for the affected K4s.
            # 91 K4s * 6 edges = 546 operations per new color.
            # C=15, so 15 * 546 = 8190 operations per step.
            # This is fast.
            
            # Let's just iterate through all C colors.
            
            for new_color in range(C):
                if new_color == old_color:
                    continue
                
                # Calculate delta violations
                delta = 0
                for k4_idx in affected_k4s:
                    e_idxs = k4_edge_indices[k4_idx]
                    # Current distinct
                    mask_old = 0
                    for e in e_idxs:
                        c = colors[e] if e != e_idx else old_color
                        mask_old |= (1 << c)
                    d_old = bin(mask_old).count('1')
                    
                    # New distinct
                    mask_new = 0
                    for e in e_idxs:
                        c = colors[e] if e != e_idx else new_color
                        mask_new |= (1 << c)
                    d_new = bin(mask_new).count('1')
                    
                    v_old = max(0, 5 - d_old)
                    v_new = max(0, 5 - d_new)
                    delta += (v_new - v_old)
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color != old_color:
                colors[e_idx] = best_new_color
                cur_viol += best_delta
                stuck_count = 0
            else:
                stuck_count += 1
            
            if cur_viol == 0:
                return colors.tolist()
            
            if stuck_count > max_stuck:
                # Restart
                colors = np.random.randint(0, C, size=num_edges)
                cur_viol = total_violations(colors)
                if cur_viol == 0:
                    return colors.tolist()
                stuck_count = 0
    
    # Fallback: return a valid coloring with 17 colors (or 16?)
    # The seed program uses (i+j) % 17.
    # Let's check if (i+j) % 16 works?
    # If we use (i+j) % 16, the colors are 0..15.
    # Is it valid?
    # Let's just return the seed fallback to be safe.
    # Actually, we can try to find a valid coloring with 16 colors quickly.
    # But the seed program's fallback is (i+j) % 17.
    # Let's stick to that.
    
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
