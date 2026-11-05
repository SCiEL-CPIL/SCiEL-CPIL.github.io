import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)

    num_k4s = len(k4s)

    def verify_valid(colors_list):
        for e_list in k4_edges:
            cols = set([colors_list[e] for e in e_list])
            if len(cols) < 5:
                return False
        return True

    def get_baseline():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    baseline = get_baseline()
    if not verify_valid(baseline):
        return [random.randint(0, 119) for _ in range(num_edges)]

    best_colors_used = 17
    best_result = baseline.copy()

    # Helper to compute delta_bad for a single edge change
    def delta_bad(e_idx, old_color, new_color, colors):
        delta = 0
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            # Get colors of the other 5 edges
            other_colors = [colors[e] for e in e_list if e != e_idx]
            
            s_before = set(other_colors)
            s_before.add(old_color)
            bad_before = 1 if len(s_before) < 5 else 0
            
            s_after = set(other_colors)
            s_after.add(new_color)
            bad_after = 1 if len(s_after) < 5 else 0
            
            delta += bad_after - bad_before
        return delta

    # Helper to count total bad K4s
    def count_bad(colors):
        bad = 0
        for e_list in k4_edges:
            if len(set(colors[e] for e in e_list)) < 5:
                bad += 1
        return bad

    # Strategy 1: Algebraic Construction using GF(4)
    # Embed 16 vertices into GF(4)^2 (or GF(4)^1 if we treat them as elements of GF(4) extended?)
    # Actually, 16 vertices can be identified with elements of GF(4)^2.
    # Let V = GF(4)^2.
    # Edge color for (u, v) can be defined via a bilinear form or similar.
    # However, the colors must be integers 0..C-1.
    # We can map GF(4)^2 to integers 0..15.
    # If we use a linear map L: GF(4)^2 -> GF(4), the image has size 4. Not enough colors.
    # If we use a quadratic form Q(u,v) = u * v (component wise?) or similar.
    # Let's try a specific construction:
    # Identify vertices with pairs (a, b) in GF(4) x GF(4).
    # Color of edge ((a1, b1), (a2, b2)) is (a1*a2 + b1*b2) mapped to int?
    # GF(4) has 4 elements. The product is in GF(4).
    # This gives at most 4 colors. We need at least 13.
    # So we need a more complex mapping.
    
    # Alternative: Use the structure of the projective plane or similar?
    # Or simply use the fact that (i+j) % 17 works, and try to compress it.
    # The baseline uses 17 colors. The lower bound is 13.
    # Let's try to map the 17 colors to 13 or 14 by merging some colors carefully.
    
    # Strategy 2: Simulated Annealing on edge colors with delta evaluation
    # This is the most robust approach given the time limit.
    
    # We will run SA for target_C in [13, 14, 15].
    # For each target_C, we try multiple restarts.
    
    def run_sa(target_C, num_restarts, max_iters):
        """
        Run Simulated Annealing to find a valid coloring with target_C colors.
        Returns (colors, bad_count) if successful or best found.
        """
        best_bad = float('inf')
        best_colors = None
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random colors
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad(colors)
            
            if current_bad == 0:
                return colors, 0
            
            T = 2.0
            cooling = 0.999
            T_min = 0.01
            no_improve = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                
                if new_color == old_color:
                    continue
                
                delta = delta_bad(e_idx, old_color, new_color, colors)
                
                if delta < 0:
                    colors[e_idx] = new_color
                    current_bad += delta
                    no_improve = 0
                elif delta == 0:
                    if random.random() < 0.5:
                        colors[e_idx] = new_color
                        no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < min(1.0, np.exp(-delta / max(T, 0.01))):
                        colors[e_idx] = new_color
                        current_bad += delta
                    no_improve += 1
                
                if current_bad == 0:
                    return colors, 0
                
                if no_improve > 100:
                    T = max(T_min, T * 0.9)
                    no_improve = 0
                else:
                    T = max(T_min, T * cooling)
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_colors = colors.copy()
        
        return best_colors, best_bad

    # Try target C = 13, 14, 15
    # Allocate time proportionally
    # 13 is hardest, give it more time? Or start with 15 to get a good incumbent, then 14, then 13?
    # Since 17 is already a valid solution, we just want to improve it.
    
    # Try 14 first as it might be easier than 13
    # Then try 13
    
    # Time budget allocation
    # Total available: ~105s
    # 15: 10s
    # 14: 30s
    # 13: 60s
    
    # Let's just loop and check time.
    
    for target_C in [14, 13]:
        if time.time() > deadline - 10:
            break
        
        # Determine number of restarts and iterations based on remaining time
        remaining_time = deadline - 10 - time.time()
        if remaining_time <= 0:
            break
            
        # Estimate time per iteration.
        # delta_bad is fast. count_bad is slow (only used at start).
        # Let's run until time runs out for this target_C, or until success.
        
        # We'll run in a loop of restarts
        # Each restart runs max_iters iterations.
        # Let's set max_iters dynamically or just run a fixed large number and break on time.
        
        max_iters = 5000
        num_restarts = 1000 # Will break early due to time
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad(colors)
            
            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors.copy()
                break
            
            T = 2.0
            cooling = 0.999
            T_min = 0.01
            no_improve = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                
                if new_color == old_color:
                    continue
                
                delta = delta_bad(e_idx, old_color, new_color, colors)
                
                if delta < 0:
                    colors[e_idx] = new_color
                    current_bad += delta
                    no_improve = 0
                elif delta == 0:
                    if random.random() < 0.5:
                        colors[e_idx] = new_color
                        no_improve = 0
                else:
                    if random.random() < min(1.0, np.exp(-delta / max(T, 0.01))):
                        colors[e_idx] = new_color
                        current_bad += delta
                    no_improve += 1
                
                if current_bad == 0:
                    if target_C < best_colors_used:
                        best_colors_used = target_C
                        best_result = colors.copy()
                    break
                
                if no_improve > 100:
                    T = max(T_min, T * 0.9)
                    no_improve = 0
                else:
                    T = max(T_min, T * cooling)
            
            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors.copy()
                break

    # Final verification
    if verify_valid(best_result):
        return best_result
    else:
        return baseline
