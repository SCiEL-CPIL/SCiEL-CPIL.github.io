"""Hypergraph Ramsey R(4,5;3) construction via fast local search with incremental updates."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin from 150s kill
    
    target_n = 34
    M = target_n * (target_n - 1) * (target_n - 2) // 6  # 5984 edges
    
    # Precompute edge indices
    edge_index = {}
    idx = 0
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                edge_index[(a, b, c)] = idx
                idx += 1
                
    # Precompute edge indices for all K4s and K5s
    k4_edges = []
    for quad in combinations(range(target_n), 4):
        edges = []
        for triple in combinations(quad, 3):
            edges.append(edge_index[triple])
        k4_edges.append(edges)
        
    k5_edges = []
    for quint in combinations(range(target_n), 5):
        edges = []
        for triple in combinations(quint, 3):
            edges.append(edge_index[triple])
        k5_edges.append(edges)
        
    k4_np = np.array(k4_edges, dtype=np.int32)
    k5_np = np.array(k5_edges, dtype=np.int32)
    
    num_k4 = len(k4_edges)
    num_k5 = len(k5_edges)
    
    # Precompute: for each edge, which K4s and K5s contain it
    # This allows O(1) delta computation when flipping an edge
    edge_to_k4s = [[] for _ in range(M)]
    for i, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(i)
            
    edge_to_k5s = [[] for _ in range(M)]
    for i, edges in enumerate(k5_edges):
        for e in edges:
            edge_to_k5s[e].append(i)
    
    # Convert to numpy arrays for faster access
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in edge_to_k4s]
    edge_to_k5s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in edge_to_k5s]
    
    def compute_objective(colors):
        k4_sums = np.sum(colors[k4_np], axis=1)
        red_k4_count = np.sum(k4_sums == 0)
        k5_sums = np.sum(colors[k5_np], axis=1)
        blue_k5_count = np.sum(k5_sums == 10)
        return red_k4_count + blue_k5_count
    
    def compute_delta(edge_idx, colors):
        """Compute change in objective when flipping edge_idx. Returns delta_obj."""
        old_val = colors[edge_idx]
        new_val = 1 - old_val
        
        delta = 0
        
        # Check K4s containing this edge
        k4_indices = edge_to_k4s_arr[edge_idx]
        if len(k4_indices) > 0:
            # Get current sums for these K4s
            for k4i in k4_indices:
                edges_in_k4 = k4_np[k4i]
                # Compute sum before and after flip
                # Only need to check if this was the last red edge or becomes red
                current_sum = np.sum(colors[edges_in_k4])
                if old_val == 0:  # Was red, becoming blue
                    # If current_sum == 0, this K4 was all red (violation), removing one red breaks it
                    if current_sum == 0:
                        delta -= 1  # Remove one violation
                else:  # Was blue, becoming red
                    # If current_sum == 3 (all other 3 are red), adding this red makes it all red
                    if current_sum == 3:
                        delta += 1  # Add one violation
        
        # Check K5s containing this edge
        k5_indices = edge_to_k5s_arr[edge_idx]
        if len(k5_indices) > 0:
            for k5i in k5_indices:
                edges_in_k5 = k5_np[k5i]
                current_sum = np.sum(colors[edges_in_k5])
                if old_val == 1:  # Was blue, becoming red
                    # If current_sum == 10, this K5 was all blue (violation), removing one blue breaks it
                    if current_sum == 10:
                        delta -= 1  # Remove one violation
                else:  # Was red, becoming blue
                    # If current_sum == 9 (all other 9 are blue), adding this blue makes it all blue
                    if current_sum == 9:
                        delta += 1  # Add one violation
        
        return delta
    
    # Initialize with a structured coloring to start closer to a solution
    # Use a simple modular pattern as initial guess
    colors = np.zeros(M, dtype=np.int8)
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                e = edge_index[(a, b, c)]
                # Simple heuristic: color based on sum mod 2
                colors[e] = (a + b + c) % 2
    
    current_obj = compute_objective(colors)
    best_obj = current_obj
    best_colors = colors.copy()
    
    # Fast local search with incremental updates
    no_improve = 0
    max_no_improve = 2000
    step = 0
    
    while time.time() - start_time < deadline - 5:
        # Pick a random edge to flip
        edge_idx = np.random.randint(0, M)
        
        # Compute delta efficiently
        delta = compute_delta(edge_idx, colors)
        
        if delta < 0:
            # Improvement: accept
            colors[edge_idx] ^= 1
            current_obj += delta
            no_improve = 0
            if current_obj < best_obj:
                best_obj = current_obj
                best_colors = colors.copy()
                if best_obj == 0:
                    return {'n': target_n, 'colors': best_colors.tolist()}
        elif delta == 0:
            # Neutral: accept with some probability to escape local minima
            if random.random() < 0.1:
                colors[edge_idx] ^= 1
                no_improve += 1
            else:
                no_improve += 1
        else:
            # Worsening: accept with small probability (simulated annealing-like)
            if random.random() < 0.01:
                colors[edge_idx] ^= 1
                current_obj += delta
                no_improve += 1
            else:
                no_improve += 1
        
        step += 1
        
        # If stuck, try a more aggressive restart
        if no_improve > max_no_improve:
            if time.time() - start_time > deadline - 10:
                break
            
            # Perturb the current solution
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                edge_idx = np.random.randint(0, M)
                colors[edge_idx] ^= 1
            current_obj = compute_objective(colors)
            no_improve = 0
            
            # If perturbation made things worse, revert to best
            if current_obj > best_obj:
                colors = best_colors.copy()
                current_obj = best_obj
    
    # Final check
    if best_obj == 0:
        return {'n': target_n, 'colors': best_colors.tolist()}
    
    # Try a few more restarts if time permits
    for restart in range(5):
        if time.time() - start_time > deadline - 3:
            break
        
        # New random initialization
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        current_obj = compute_objective(colors)
        
        # Quick local search
        for _ in range(5000):
            if time.time() - start_time > deadline - 3:
                break
            edge_idx = np.random.randint(0, M)
            delta = compute_delta(edge_idx, colors)
            if delta < 0:
                colors[edge_idx] ^= 1
                current_obj += delta
                if current_obj == 0:
                    return {'n': target_n, 'colors': colors.tolist()}
            elif delta == 0 and random.random() < 0.05:
                colors[edge_idx] ^= 1
    
    # Fallback: return best found
    if best_obj == 0:
        return {'n': target_n, 'colors': best_colors.tolist()}
    
    # Return a valid smaller solution as fallback
    # n=9 with modular coloring is known to work
    n_fallback = 9
    fallback_colors = []
    for a in range(n_fallback):
        for b in range(a + 1, n_fallback):
            for c in range(b + 1, n_fallback):
                fallback_colors.append(0 if (a + b + c) % 3 == 0 else 1)
    return {'n': n_fallback, 'colors': fallback_colors}
# EVOLVE-BLOCK-END
