import time
import random
import numpy as np

N = 16
NUM_EDGES = N*(N-1)//2

edge_map = {}
idx = 0
for i in range(N):
    for j in range(i+1, N):
        edge_map[(i,j)] = idx
        idx += 1

cycles = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Cycle 1
                e1 = edge_map[(a,b)]
                e2 = edge_map[(b,c)]
                e3 = edge_map[(c,d)]
                e4 = edge_map[(d,a)]
                c_id = len(cycles)
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(c_id)
                edge_to_cycles[e2].append(c_id)
                edge_to_cycles[e3].append(c_id)
                edge_to_cycles[e4].append(c_id)
                
                # Cycle 2
                e1 = edge_map[(a,b)]
                e2 = edge_map[(b,d)]
                e3 = edge_map[(d,c)]
                e4 = edge_map[(c,a)]
                c_id = len(cycles)
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(c_id)
                edge_to_cycles[e2].append(c_id)
                edge_to_cycles[e3].append(c_id)
                edge_to_cycles[e4].append(c_id)
                
                # Cycle 3
                e1 = edge_map[(a,c)]
                e2 = edge_map[(c,b)]
                e3 = edge_map[(b,d)]
                e4 = edge_map[(d,a)]
                c_id = len(cycles)
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(c_id)
                edge_to_cycles[e2].append(c_id)
                edge_to_cycles[e3].append(c_id)
                edge_to_cycles[e4].append(c_id)

def is_violated(c_id, colors):
    e1, e2, e3, e4 = cycles[c_id]
    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
    return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    colors = [0] * NUM_EDGES
    # violation_status[i] is True if cycle i is violated
    violation_status = [False] * len(cycles)
    current_violations = 0
    current_distinct = 1
    best_colors = colors.copy()
    best_distinct = 1
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
            
        # Update violations for cycles involving e
        delta_v = 0
        affected_cycles = edge_to_cycles[e]
        
        # Check current status of affected cycles
        # We need to know if they were violated before, and if they will be violated after.
        # Since we are changing colors[e], only these cycles are affected.
        
        # Store old status
        old_statuses = [violation_status[c_id] for c_id in affected_cycles]
        
        colors[e] = new_color
        
        new_statuses = []
        for c_id in affected_cycles:
            v = is_violated(c_id, colors)
            new_statuses.append(v)
            violation_status[c_id] = v
            
        # Calculate delta
        for i, c_id in enumerate(affected_cycles):
            if old_statuses[i] and not new_statuses[i]:
                delta_v -= 1
            elif not old_statuses[i] and new_statuses[i]:
                delta_v += 1
                
        new_violations = current_violations + delta_v
        
        # Decision
        if new_violations == 0:
            # Valid
            new_distinct = len(set(colors))
            if new_distinct > current_distinct:
                current_distinct = new_distinct
                best_colors = colors.copy()
                best_distinct = current_distinct
            elif new_distinct < current_distinct:
                # Revert
                colors[e] = old_color
                for i, c_id in enumerate(affected_cycles):
                    violation_status[c_id] = old_statuses[i]
                continue
            else:
                # Accept random walk
                pass
        else:
            # Invalid
            if new_violations > current_violations:
                prob = np.exp(-(new_violations - current_violations) / T)
                if random.random() < prob:
                    current_violations = new_violations
                else:
                    # Revert
                    colors[e] = old_color
                    for i, c_id in enumerate(affected_cycles):
                        violation_status[c_id] = old_statuses[i]
                    continue
            else:
                current_violations = new_violations
                if current_violations == 0:
                    new_distinct = len(set(colors))
                    if new_distinct > best_distinct:
                        best_distinct = new_distinct
                        best_colors = colors.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_colors
