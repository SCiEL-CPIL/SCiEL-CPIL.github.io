import time
import random
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]
    
    # Try to find a valid coloring for n from 34 down to 10
    # We prioritize larger n.
    # We will try a few random restarts for each n.
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Precompute structures for this n
        # This might be slow for large n if done repeatedly, but we only do it once per n.
        try:
            edges = list(itertools.combinations(range(n), 3))
            num_edges = len(edges)
            edge_index = {e: i for i, e in enumerate(edges)}
            
            sets4 = list(itertools.combinations(range(n), 4))
            sets5 = list(itertools.combinations(range(n), 5))
            num_sets4 = len(sets4)
            num_sets5 = len(sets5)
            
            # Map edge to list of 4-set indices and 5-set indices
            # This is the most expensive part.
            # For each 4-set, it contains 4 edges.
            edge_to_4sets = [[] for _ in range(num_edges)]
            for s4_idx, s4 in enumerate(sets4):
                # s4 is a tuple of 4 vertices
                # Generate 4 triples from s4
                for i in range(4):
                    # triple is s4 with s4[i] removed
                    triple = s4[:i] + s4[i+1:]
                    e_idx = edge_index[triple]
                    edge_to_4sets[e_idx].append(s4_idx)
                    
            edge_to_5sets = [[] for _ in range(num_edges)]
            for s5_idx, s5 in enumerate(sets5):
                # s5 is a tuple of 5 vertices
                # Generate 10 triples from s5
                for triple in itertools.combinations(s5, 3):
                    e_idx = edge_index[triple]
                    edge_to_5sets[e_idx].append(s5_idx)
                    
        except Exception:
            # If precomputation fails or is too slow/memory heavy, skip this n
            continue

        # Try to find a valid coloring with local search
        found = False
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize colors
            # Use a heuristic or random
            # Random is fine, but maybe biased?
            colors = [random.randint(0, 1) for _ in range(num_edges)]
            
            # Initialize violation counts
            # red_count_4[s4_idx] = number of red edges in s4
            red_count_4 = [0] * num_sets4
            # blue_count_5[s5_idx] = number of blue edges in s5
            blue_count_5 = [0] * num_sets5
            
            # Populate counts
            for e_idx, color in enumerate(colors):
                if color == 0: # Red
                    for s4_idx in edge_to_4sets[e_idx]:
                        red_count_4[s4_idx] += 1
                else: # Blue
                    for s5_idx in edge_to_5sets[e_idx]:
                        blue_count_5[s5_idx] += 1
                        
            # Calculate initial violations
            viol4 = sum(1 for c in red_count_4 if c == 4)
            viol5 = sum(1 for c in blue_count_5 if c == 10)
            total_viol = viol4 + viol5
            
            if total_viol == 0:
                found = True
                break
                
            # Local Search
            # Simulated Annealing or Hill Climbing
            # We'll use a simple hill climbing with random restarts if stuck
            current_temp = 1.0
            cooling = 0.99
            
            # To speed up, we can just do hill climbing (accept only if improvement)
            # If stuck, restart.
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Shuffle edges to avoid bias? Or just iterate?
                # Iterating over all edges is num_edges * cost_per_edge
                # cost_per_edge is ~500 updates.
                # 6000 * 500 = 3M ops per full pass.
                # This is fast enough for a few passes.
                
                for e_idx in range(num_edges):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Try flipping e_idx
                    old_color = colors[e_idx]
                    new_color = 1 - old_color
                    
                    delta_viol = 0
                    
                    if old_color == 0: # Was Red, now Blue
                        # Red count decreases in 4-sets
                        for s4_idx in edge_to_4sets[e_idx]:
                            old_c = red_count_4[s4_idx]
                            new_c = old_c - 1
                            if old_c == 4:
                                delta_viol -= 1
                            if new_c == 4:
                                delta_viol += 1
                            # Update count for next step (or rollback)
                            # We'll update in place and rollback if rejected
                            red_count_4[s4_idx] = new_c
                            
                        # Blue count increases in 5-sets
                        for s5_idx in edge_to_5sets[e_idx]:
                            old_c = blue_count_5[s5_idx]
                            new_c = old_c + 1
                            if old_c == 10:
                                delta_viol -= 1
                            if new_c == 10:
                                delta_viol += 1
                            blue_count_5[s5_idx] = new_c
                            
                    else: # Was Blue, now Red
                        # Blue count decreases in 5-sets
                        for s5_idx in edge_to_5sets[e_idx]:
                            old_c = blue_count_5[s5_idx]
                            new_c = old_c - 1
                            if old_c == 10:
                                delta_viol -= 1
                            if new_c == 10:
                                delta_viol += 1
                            blue_count_5[s5_idx] = new_c
                            
                        # Red count increases in 4-sets
                        for s4_idx in edge_to_4sets[e_idx]:
                            old_c = red_count_4[s4_idx]
                            new_c = old_c + 1
                            if old_c == 4:
                                delta_viol -= 1
                            if new_c == 4:
                                delta_viol += 1
                            red_count_4[s4_idx] = new_c
                            
                    total_viol += delta_viol
                    
                    if total_viol < 0 or (total_viol == 0 and delta_viol < 0): # Strict improvement
                        colors[e_idx] = new_color
                        improved = True
                        if total_viol == 0:
                            found = True
                            break
                    else:
                        # Rollback
                        if old_color == 0:
                            for s4_idx in edge_to_4sets[e_idx]:
                                red_count_4[s4_idx] += 1
                            for s5_idx in edge_to_5sets[e_idx]:
                                blue_count_5[s5_idx] -= 1
                        else:
                            for s5_idx in edge_to_5sets[e_idx]:
                                blue_count_5[s5_idx] += 1
                            for s4_idx in edge_to_4sets[e_idx]:
                                red_count_4[s4_idx] -= 1
                        total_viol -= delta_viol
                        
                if found:
                    break
                    
            if found:
                best_n = n
                best_colors = colors
                break
                
    return {'n': best_n, 'colors': best_colors}
