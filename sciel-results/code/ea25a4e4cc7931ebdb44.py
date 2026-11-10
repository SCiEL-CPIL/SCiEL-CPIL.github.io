import numpy as np
import time
import random

def construct():
    # Precompute K4 edge indices
    # Vertices 0..15
    # Edges (i,j) with i<j
    # Map (i,j) to index
    # Index = i*15 - i*(i-1)//2 + (j-i-1) ? 
    # Simpler: generate list
    edges = []
    idx_map = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            idx_map[(i,j)] = idx
            edges.append((i,j))
            idx += 1
            
    k4s = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    k4s.append((a,b,c,d))
                    
    k4_edge_indices = []
    for k4 in k4s:
        a,b,c,d = k4
        e1 = idx_map[(a,b)]
        e2 = idx_map[(a,c)]
        e3 = idx_map[(a,d)]
        e4 = idx_map[(b,c)]
        e5 = idx_map[(b,d)]
        e6 = idx_map[(c,d)]
        k4_edge_indices.append([e1, e2, e3, e4, e5, e6])
        
    k4_edge_indices = np.array(k4_edge_indices, dtype=int)
    num_k4s = len(k4_edge_indices)
    
    def get_violations(colors):
        # colors: np.array of length 120
        # k4_edge_indices: shape (1820, 6)
        k4_colors = colors[k4_edge_indices]
        # Count matches
        # c1: (1820, 6, 1), c2: (1820, 1, 6)
        c1 = k4_colors[:, :, None]
        c2 = k4_colors[:, None, :]
        matches = (c1 == c2)
        total_matches = np.sum(matches, axis=(1, 2))
        # Valid if total_matches <= 8
        # Violation if total_matches > 8
        violations = np.where(total_matches > 8)[0]
        return violations

    def try_find_valid(k, time_limit):
        start_time = time.time()
        best_violations = 10**9
        best_colors = None
        
        # Try a few restarts
        for restart in range(20):
            if time.time() > start_time + time_limit:
                break
                
            colors = np.random.randint(0, k, 120)
            violations = get_violations(colors)
            current_violations = len(violations)
            
            if current_violations == 0:
                return colors
                
            no_improve_count = 0
            max_no_improve = 5000
            
            while current_violations > 0:
                if time.time() > start_time + time_limit:
                    break
                    
                # Pick a random violation
                v_idx = violations[np.random.randint(len(violations))]
                e_indices = k4_edge_indices[v_idx]
                e_idx = e_indices[np.random.randint(6)]
                
                old_color = colors[e_idx]
                new_color = np.random.randint(0, k)
                
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                new_violations = get_violations(colors)
                new_count = len(new_violations)
                
                if new_count < current_violations:
                    violations = new_violations
                    current_violations = new_count
                    no_improve_count = 0
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors.copy()
                        if current_violations == 0:
                            return colors
                else:
                    # Revert
                    colors[e_idx] = old_color
                    no_improve_count += 1
                    
                    # If stuck, try a random move (simulated annealing style or just random walk)
                    if no_improve_count > max_no_improve:
                        # Random restart within this run? Or just accept neutral?
                        # Let's try accepting neutral moves to escape
                        if new_count == current_violations and np.random.random() < 0.1:
                            violations = new_violations
                            no_improve_count = 0
                        else:
                            # Reset no_improve to allow more tries
                            no_improve_count = max(0, no_improve_count - 100)

        return None

    deadline = time.time() + 110
    
    # Try 15 colors
    result = try_find_valid(15, 30)
    if result is not None:
        # Check if we can do better with 14?
        # Only if we have time
        if time.time() < deadline - 20:
            result14 = try_find_valid(14, 10)
            if result14 is not None:
                result = result14
            # Try 13?
            if time.time() < deadline - 5:
                result13 = try_find_valid(13, 5)
                if result13 is not None:
                    result = result13
                    
        # Convert to list of ints
        return [int(x) for x in result]
        
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
