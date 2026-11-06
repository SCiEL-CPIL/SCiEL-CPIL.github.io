import time
import random
import itertools

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initialize with n-1 colors
    # Edge (i, j) with i < j gets color i
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            colors[idx] = i
            idx += 1
            
    # Precompute cycles
    # For each 4-subset, store the 3 cycles as lists of edge indices
    # Edge index for (u, v) with u < v:
    # We can create a lookup table edge_idx[u][v]
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # List of all 4-subsets and their 3 cycles
    # cycles[subset] = [ [e1, e2, e3, e4], [e1, e2, e3, e4], [e1, e2, e3, e4] ]
    all_cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c1 = [edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]]
        # Cycle 2: a-b-d-c-a
        c2 = [edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]]
        # Cycle 3: a-c-b-d-a
        c3 = [edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]]
        all_cycles.append((quad, [c1, c2, c3]))
        
    # Map edge index to list of cycles containing it
    edge_to_cycles = [[] for _ in range(N*(N-1)//2)]
    for i, (quad, cycles) in enumerate(all_cycles):
        for cycle in cycles:
            for e in cycle:
                edge_to_cycles[e].append(i)
                
    def is_rainbow(cycle, cols):
        c1 = cols[cycle[0]]
        c2 = cols[cycle[1]]
        c3 = cols[cycle[2]]
        c4 = cols[cycle[3]]
        return len({c1, c2, c3, c4}) == 4
        
    def check_validity(cols):
        for i, (quad, cycles) in enumerate(all_cycles):
            for cycle in cycles:
                if is_rainbow(cycle, cols):
                    return False
        return True
        
    # Local Search
    best_cols = colors[:]
    best_count = len(set(best_cols))
    current_cols = colors[:]
    current_count = best_count
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, N*(N-1)//2 - 1)
        old_color = current_cols[e]
        
        # Try a new color
        # Sometimes try an existing color, sometimes a new one
        if random.random() < 0.5:
            new_color = random.randint(0, 119)
        else:
            # Try to introduce a new color not in use
            used = set(current_cols)
            available = [c for c in range(120) if c not in used]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, 119)
                
        if new_color == old_color:
            continue
            
        current_cols[e] = new_color
        
        # Check validity
        valid = True
        for cycle_idx in edge_to_cycles[e]:
            cycles = all_cycles[cycle_idx][1]
            for cycle in cycles:
                if e in cycle: # Only check if this edge is in the cycle
                    if is_rainbow(cycle, current_cols):
                        valid = False
                        break
            if not valid:
                break
                
        if valid:
            new_count = len(set(current_cols))
            if new_count > current_count:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_cols = current_cols[:]
            else:
                # Accept with some probability to escape local optima
                # Simple acceptance: always accept if valid, even if count doesn't increase?
                # This might lead to drifting.
                # Let's accept if valid.
                pass
        else:
            current_cols[e] = old_color
            
    return best_cols
