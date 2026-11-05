import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and which edges they contain
    k4s = []
    k4_edge_indices = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    e1 = edge_idx[(a, b)]
                    e2 = edge_idx[(a, c)]
                    e3 = edge_idx[(a, d)]
                    e4 = edge_idx[(b, c)]
                    e5 = edge_idx[(b, d)]
                    e6 = edge_idx[(c, d)]
                    k4s.append((a, b, c, d))
                    k4_edge_indices.append([e1, e2, e3, e4, e5, e6])
    
    # Initialize with 17-coloring
    colors = np.array([(i + j) % 17 for i, j in edges], dtype=int)
    
    # Function to count violations and max color
    def get_stats(c):
        max_c = c.max() + 1
        violations = 0
        for e_idx in k4_edge_indices:
            cc = c[e_idx]
            if len(set(cc)) < 5:
                violations += 1
        return max_c, violations
    
    best_colors = colors.copy()
    best_score = get_stats(best_colors)[0] + 100 * get_stats(best_colors)[1]
    
    # Local Search
    current_colors = colors.copy()
    current_max_c, current_viol = get_stats(current_colors)
    current_score = current_max_c + 100 * current_viol
    
    # We want to minimize colors. 
    # Strategy: Try to reduce the max color used.
    # Iterate through possible max colors from 17 down to 13.
    
    # A more robust approach: Simulated Annealing on the score.
    # But we want to specifically target lower color counts.
    
    # Let's try a greedy reduction.
    # Try to map colors 0..16 to 0..12.
    
    # Actually, let's just run a local search minimizing (max_color, violations).
    # Lexicographic order: minimize max_color first, then violations.
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, len(edges)-1)
        old_c = current_colors[e]
        
        # Try a new color
        # Bias towards lower colors
        new_c = random.randint(0, 16)
        if new_c == old_c:
            continue
            
        current_colors[e] = new_c
        
        new_max_c = max(current_colors.max(), new_c)
        # Note: if old_c was the max and we reduced it, max might drop.
        # But calculating max every time is O(120).
        # We can maintain a count of colors.
        
        # For simplicity in this short script, we just recalculate stats if the move looks promising.
        # But checking all 1820 K4s is slow.
        
        # Optimization: Only check K4s affected by edge e.
        # We need to know which K4s contain edge e.
        # Let's precompute this.
        
        # Re-initializing precomputation for speed
        # Actually, let's just use a simpler heuristic:
        # Try to force colors into a smaller range.
        
        # Break if time is up
        if time.time() > deadline - 5:
            break

    # If we didn't find a better one, return the initial 17-coloring.
    # To ensure we beat 1.0, we need < 17 colors.
    # If local search is too slow, we might just return 17.
    # But 17 is the seed. We need improvement.
    
    # Let's try a specific known 13-coloring pattern if possible.
    # If not, return best found.
    
    # Since I cannot guarantee the local search finds 13 in 140s with simple Python,
    # I will return the 17-coloring if no improvement is found, 
    # BUT I will try a very fast greedy pass first.
    
    # Greedy Pass: Try to reduce max color.
    # For each edge, try to assign the smallest color that doesn't increase violations too much.
    
    # Given the complexity, I will return the seed coloring if no fast improvement is found,
    # but I will add a check: if we can find a 16-coloring quickly, we return it.
    
    # Actually, the prompt says "Minimize the number of colours used".
    # If I return 17, score is 1.0.
    # If I return 16, score is 17/16 = 1.0625.
    
    # I will implement a fast local search that specifically tries to reduce the max color.
    
    # Reset for a proper local search implementation
    current_colors = colors.copy()
    # Precompute K4s containing each edge
    edge_to_k4s = [[] for _ in range(len(edges))]
    for k, e_list in enumerate(k4_edge_indices):
        for e in e_list:
            edge_to_k4s[e].append(k)
    
    # Cache violation count per K4
    k4_violations = np.zeros(len(k4s), dtype=int)
    for k, e_list in enumerate(k4_edge_indices):
        if len(set(current_colors[e_list])) < 5:
            k4_violations[k] = 1
    
    total_violations = np.sum(k4_violations)
    max_color = current_colors.max()
    
    best_total_violations = total_violations
    best_max_color = max_color
    best_colors = current_colors.copy()
    
    t = 10.0
    step = 0
    
    while time.time() < deadline:
        step += 1
        e = random.randint(0, len(edges)-1)
        old_c = current_colors[e]
        new_c = random.randint(0, max_color) # Try within current range
        
        if new_c == old_c:
            continue
            
        # Calculate delta violations
        delta_viol = 0
        for k in edge_to_k4s[e]:
            e_list = k4_edge_indices[k]
            old_cc = current_colors[e_list]
            old_count = len(set(old_cc))
            
            # Simulate change
            # Find index of e in e_list
            idx = e_list.index(e)
            new_cc = list(old_cc)
            new_cc[idx] = new_c
            new_count = len(set(new_cc))
            
            old_v = 1 if old_count < 5 else 0
            new_v = 1 if new_count < 5 else 0
            delta_viol += new_v - old_v
            
        new_total_violations = total_violations + delta_viol
        new_max_color = max(max_color, new_c)
        # If old_c was the unique max, new_max might be lower.
        if old_c == max_color and np.all(current_colors != old_c) == False: # Check if any other edge has old_c
             # This check is O(N). We can maintain a count array.
             pass
             
        # Simple acceptance: Accept if violations decrease, or if violations same and max color decreases.
        # Or use SA.
        
        # To keep it fast, let's just accept if it improves the lexicographic score (max_color, violations)
        # Or if it's a random walk to escape local minima.
        
        if new_total_violations < total_violations or (new_total_violations == total_violations and new_max_color < max_color):
            current_colors[e] = new_c
            total_violations = new_total_violations
            max_color = new_max_color
            if total_violations < best_total_violations or (total_violations == best_total_violations and max_color < best_max_color):
                best_total_violations = total_violations
                best_max_color = max_color
                best_colors = current_colors.copy()
        else:
            # SA acceptance
            delta_score = (new_max_color - max_color) * 100 + (new_total_violations - total_violations)
            if random.random() < np.exp(-delta_score / t):
                current_colors[e] = new_c
                total_violations = new_total_violations
                max_color = new_max_color
                if total_violations < best_total_violations or (total_violations == best_total_violations and max_color < best_max_color):
                    best_total_violations = total_violations
                    best_max_color = max_color
                    best_colors = current_colors.copy()
        
        t *= 0.999
        if t < 0.1:
            t = 0.1

    # Return best found
    # If best is 17, return seed.
    return best_colors.tolist()
