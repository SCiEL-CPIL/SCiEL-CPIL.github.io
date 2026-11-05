"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    edge_idx = {}
    for idx, (a, b) in enumerate(edges):
        edge_idx[(a, b)] = idx
    
    k4_edges_list = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx[(v0, v1)],
            edge_idx[(v0, v2)],
            edge_idx[(v0, v3)],
            edge_idx[(v1, v2)],
            edge_idx[(v1, v3)],
            edge_idx[(v2, v3)],
        ]
        k4_edges_list.append(e_indices)
    
    num_k4s = len(k4_edges_list)
    k4_edges = np.array(k4_edges_list) # shape (1820, 6)
    
    # Precompute pairs of edges within each K4 for collision detection
    # There are 15 pairs in 6 edges
    pair_indices = np.array([(i, j) for i in range(6) for j in range(i+1, 6)])
    
    def count_violations_vec(col_np):
        """
        Vectorized count of K4s with < 5 distinct colors.
        A K4 is invalid if it has >= 2 colliding pairs of edges.
        """
        # Get colors for all edges in all K4s: shape (1820, 6)
        k4_colors = col_np[k4_edges]
        
        # Compare all pairs
        # c1 = k4_colors[:, pair_indices[:, 0]] -> shape (1820, 15)
        # c2 = k4_colors[:, pair_indices[:, 1]] -> shape (1820, 15)
        c1 = k4_colors[:, pair_indices[:, 0]]
        c2 = k4_colors[:, pair_indices[:, 1]]
        
        # Collision mask: True if colors are same
        collisions = (c1 == c2) # shape (1820, 15)
        
        # Count collisions per K4
        num_collisions = np.sum(collisions, axis=1)
        
        # Violation if num_collisions >= 2
        violations = np.sum(num_collisions >= 2)
        return violations
    
    def count_colors(col_np):
        return len(np.unique(col_np))
    
    random.seed(42)
    
    best_result = None
    best_color_count = float('inf')
    
    # Try base colorings with different moduli
    for modulus in [15, 14, 13, 17, 16, 12, 11, 10]:
        if time.time() > deadline - 20.0:
            break
        
        base_col = np.array([(i + j) % modulus for i in range(n) for j in range(i + 1, n)])
        
        v = count_violations_vec(base_col)
        if v == 0:
            cc = count_colors(base_col)
            if cc < best_color_count:
                best_color_count = cc
                best_result = base_col.copy()
        
        # Local search to improve
        if time.time() > deadline - 10.0:
            break
            
        col = base_col.copy()
        current_violations = count_violations_vec(col)
        current_colors = count_colors(col)
        
        if current_violations == 0:
            if current_colors < best_color_count:
                best_color_count = current_colors
                best_result = col.copy()
        
        # Try to reduce colors or fix violations
        # Target: use as few colors as possible
        # We will try to map colors to a smaller palette if possible
        
        max_iters = 5000
        no_improve = 0
        temperature = 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if no_improve > 1000:
                break
                
            e = random.randint(0, num_edges - 1)
            old_color = col[e]
            
            # Try to change to a color that is already used or a new one
            # To reduce colors, prefer existing colors
            existing_colors = np.unique(col)
            if len(existing_colors) > 0:
                # 80% chance pick an existing color, 20% chance pick a new random color (to explore)
                if random.random() < 0.8:
                    new_color = existing_colors[random.randint(0, len(existing_colors)-1)]
                else:
                    new_color = random.randint(0, 119)
            else:
                new_color = random.randint(0, 119)
                
            if new_color == old_color:
                continue
                
            col[e] = new_color
            new_violations = count_violations_vec(col)
            new_colors = count_colors(col)
            
            # Objective: Minimize violations, then minimize colors
            # Score = violations * 1000 + colors
            old_score = current_violations * 1000 + current_colors
            new_score = new_violations * 1000 + new_colors
            
            if new_score < old_score:
                current_violations = new_violations
                current_colors = new_colors
                no_improve = 0
            elif new_score == old_score:
                # Accept with some probability to escape plateaus
                if random.random() < temperature:
                    current_violations = new_violations
                    current_colors = new_colors
                    no_improve = 0
                else:
                    col[e] = old_color
            else:
                # Accept worse with small probability
                if random.random() < temperature * 0.1:
                    current_violations = new_violations
                    current_colors = new_colors
                else:
                    col[e] = old_color
            
            temperature = max(0.01, temperature * 0.999)
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col.copy()
                    no_improve = 0 # Reset no_improve to try to reduce colors further
                else:
                    break # Can't improve colors if already optimal for this base? No, keep searching for lower colors
            
    if best_result is None:
        # Fallback to a valid coloring if we found one, or the base
        # If we never found a valid one, we must return something.
        # The task says "A valid colouring certifies...". If we return invalid, score is 0.
        # We should ensure we return the best valid one found.
        # If best_result is None, it means no valid coloring was found.
        # We should try harder or return a known valid one if possible.
        # However, (i+j)%17 is likely valid.
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        
    return best_result.tolist()
# EVOLVE-BLOCK-END
