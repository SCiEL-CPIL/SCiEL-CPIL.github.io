import time
import random
import math

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Targets n=26. Falls back to n=11 if not found in time.
    """
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Try to find a solution for n=26
    result = try_solve(26, deadline)
    if result is not None:
        return result
    
    # Fallback to known valid n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_solve(n, deadline):
    """
    Attempts to find a valid 5-coloring of K_n.
    Returns dict if found, None otherwise.
    """
    if n < 2:
        return None
        
    # Precomputation
    num_edges = n * (n - 1) // 2
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(idx_to_edge)
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            
    # Find all C4s
    # A C4 is a set of 4 vertices {a,b,c,d} and a specific cycle.
    # For each set of 4 vertices, there are 3 distinct C4s.
    c4_edges_list = [] # List of lists of edge indices
    edge_to_c4s = [[] for _ in range(num_edges)]
    
    # Generate all C4s
    # Vertices a < b < c < d
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(b, c)]
                    e3 = edge_to_idx[(c, d)]
                    e4 = edge_to_idx[(a, d)]
                    c4_id = len(c4_edges_list)
                    c4_edges_list.append([e1, e2, e3, e4])
                    edge_to_c4s[e1].append(c4_id)
                    edge_to_c4s[e2].append(c4_id)
                    edge_to_c4s[e3].append(c4_id)
                    edge_to_c4s[e4].append(c4_id)
                    
                    # Cycle 2: a-b, b-d, d-c, c-a
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(b, d)]
                    e3 = edge_to_idx[(c, d)]
                    e4 = edge_to_idx[(a, c)]
                    c4_id = len(c4_edges_list)
                    c4_edges_list.append([e1, e2, e3, e4])
                    edge_to_c4s[e1].append(c4_id)
                    edge_to_c4s[e2].append(c4_id)
                    edge_to_c4s[e3].append(c4_id)
                    edge_to_c4s[e4].append(c4_id)
                    
                    # Cycle 3: a-c, c-b, b-d, d-a
                    e1 = edge_to_idx[(a, c)]
                    e2 = edge_to_idx[(b, c)]
                    e3 = edge_to_idx[(b, d)]
                    e4 = edge_to_idx[(a, d)]
                    c4_id = len(c4_edges_list)
                    c4_edges_list.append([e1, e2, e3, e4])
                    edge_to_c4s[e1].append(c4_id)
                    edge_to_c4s[e2].append(c4_id)
                    edge_to_c4s[e3].append(c4_id)
                    edge_to_c4s[e4].append(c4_id)
    
    num_c4s = len(c4_edges_list)
    
    # Objective: Minimize count of monochromatic C4s
    # State: colors array
    # We use Simulated Annealing
    
    def get_initial_colors():
        return [random.randint(0, 4) for _ in range(num_edges)]
    
    def count_mono_c4s(colors):
        count = 0
        for c4 in c4_edges_list:
            c0 = colors[c4[0]]
            if c0 == colors[c4[1]] and c0 == colors[c4[2]] and c0 == colors[c4[3]]:
                count += 1
        return count
        
    best_colors = None
    best_energy = float('inf')
    
    # Restart strategy
    # We don't know how many restarts fit, so we run until deadline.
    # Each SA run has a fixed number of steps? Or time-based?
    # Time-based is safer.
    
    while time.time() < deadline - 5: # Leave 5s for finalization
        colors = get_initial_colors()
        current_energy = count_mono_c4s(colors)
        
        if current_energy == 0:
            return {'n': n, 'colors': colors}
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_colors = colors[:]
            
        # SA Parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        steps_per_restart = 50000 # Adjust based on speed
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            # Calculate Delta Energy
            # Only C4s containing e_idx are affected
            delta = 0
            c4s_to_check = edge_to_c4s[e_idx]
            
            for c4_id in c4s_to_check:
                c4_edges = c4_edges_list[c4_id]
                # Find the other 3 edges
                # c4_edges contains e_idx
                # Check if the C4 was mono before
                # And if it is mono after
                
                # Optimization: Pre-store the other 3 edges?
                # For now, just check the 4 edges.
                
                # Old state
                was_mono = (colors[c4_edges[0]] == colors[c4_edges[1]] == colors[c4_edges[2]] == colors[c4_edges[3]])
                
                # Apply change temporarily
                colors[e_idx] = new_color
                
                # New state
                is_mono = (colors[c4_edges[0]] == colors[c4_edges[1]] == colors[c4_edges[2]] == colors[c4_edges[3]])
                
                # Revert
                colors[e_idx] = old_color
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            # Acceptance
            if delta <= 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_color
                current_energy += delta
                
                if current_energy == 0:
                    return {'n': n, 'colors': colors}
                    
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
                
    # After loop, check best
    if best_colors is not None and count_mono_c4s(best_colors) == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
