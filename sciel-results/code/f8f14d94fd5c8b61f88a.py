import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Map (i, j) to edge index
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        
    # Precompute K4s and their edge indices
    k4_edges = []
    for quad in combinations(range(n), 4):
        k_edges = []
        for a, b in combinations(quad, 2):
            k_edges.append(edge_idx_map[(a, b)])
        k4_edges.append(k_edges)
        
    num_k4s = len(k4_edges)
    
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k_edges in enumerate(k4_edges):
        for e in k_edges:
            edge_to_k4s[e].append(k4_idx)
            
    # State variables for backtracking
    colors = [-1] * num_edges
    k4_color_counts = [{} for _ in range(num_k4s)]
    k4_unassigned = [6] * num_k4s
    
    def reset_state():
        for i in range(num_edges):
            colors[i] = -1
        for i in range(num_k4s):
            k4_color_counts[i] = {}
            k4_unassigned[i] = 6

    def assign(edge_idx, color):
        colors[edge_idx] = color
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_counts[k4_idx][color] = k4_color_counts[k4_idx].get(color, 0) + 1
            k4_unassigned[k4_idx] -= 1
            # Check constraint: distinct colors + unassigned >= 5
            distinct = len([c for c, cnt in k4_color_counts[k4_idx].items() if cnt > 0])
            if distinct + k4_unassigned[k4_idx] < 5:
                return False
        return True

    def unassign(edge_idx):
        color = colors[edge_idx]
        colors[edge_idx] = -1
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_counts[k4_idx][color] -= 1
            if k4_color_counts[k4_idx][color] == 0:
                del k4_color_counts[k4_idx][color]
            k4_unassigned[k4_idx] += 1

    def search(edge_idx, max_color_used, K):
        if time.time() > deadline:
            return False
        
        if edge_idx == num_edges:
            return True
        
        u, v = edges[edge_idx]
        
        # Symmetry breaking and candidate generation
        if edge_idx == 0:
            candidates = [0]
        else:
            candidates = []
            # Try colors 0..K-1
            # Constraint: if c > max_color_used, c must be max_color_used + 1
            for c in range(K):
                if c <= max_color_used:
                    candidates.append(c)
                elif c == max_color_used + 1:
                    candidates.append(c)
            
            # Heuristic: Try colors that are already used in adjacent K4s?
            # For now, just try in order.
            
        for c in candidates:
            if assign(edge_idx, c):
                new_max = max(max_color_used, c)
                if search(edge_idx + 1, new_max, K):
                    return True
            unassign(edge_idx)
            
        return False

    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        reset_state()
        
        # Force (0,1) to 0
        if assign(0, 0):
            if search(1, 0, K):
                # Found solution
                return [int(c) for c in colors]
        unassign(0)
        
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
