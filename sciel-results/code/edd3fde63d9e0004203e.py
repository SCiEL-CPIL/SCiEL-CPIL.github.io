import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Cache for precomputed cycles
    cycle_cache = {}
    
    for n in range(18, 8, -1):
        if time.time() > deadline - 10:
            break
            
        # Precompute cycles if not cached
        if n not in cycle_cache:
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            num_edges = len(edges)
            edge_index = {e: idx for idx, e in enumerate(edges)}
            
            cycles = []
            edge_to_cycles = [[] for _ in range(num_edges)]
            seen_cycles = set()
            
            vertices = list(range(n))
            for verts in itertools.combinations(vertices, 6):
                v = list(verts)
                for perm in itertools.permutations(v[1:]):
                    cycle_verts = [v[0]] + list(perm)
                    cycle_edges_idx = []
                    for k in range(6):
                        u = cycle_verts[k]
                        w = cycle_verts[(k+1)%6]
                        if u > w: u, w = w, u
                        cycle_edges_idx.append(edge_index[(u,w)])
                    
                    key = tuple(sorted(cycle_edges_idx))
                    if key in seen_cycles:
                        continue
                    seen_cycles.add(key)
                    
                    cycle_idx = len(cycles)
                    cycles.append(cycle_edges_idx)
                    for e_idx in cycle_edges_idx:
                        edge_to_cycles[e_idx].append(cycle_idx)
                        
            cycle_cache[n] = (edges, edge_index, cycles, edge_to_cycles)
            
        edges, edge_index, cycles, edge_to_cycles = cycle_cache[n]
        num_edges = len(edges)
        num_cycles = len(cycles)
        
        # Try multiple restarts
        found_solution = None
        attempts = 0
        max_attempts = 30
        
        while time.time() < deadline - 5 and attempts < max_attempts:
            attempt_deadline = time.time() + 4
            attempts += 1
            
            sol = local_search(n, edges, edge_index, cycles, edge_to_cycles, num_edges, num_cycles, attempt_deadline)
            if sol is not None:
                found_solution = sol
                break
                
        if found_solution is not None:
            return found_solution
            
    # Fallback
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def local_search(n, edges, edge_index, cycles, edge_to_cycles, num_edges, num_cycles, deadline):
    if num_cycles == 0:
        return {'n': n, 'colors': [0]*num_edges}

    # Initialize random colors
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Initialize counts
    counts = [[0]*4 for _ in range(num_cycles)]
    for cy_idx, cy_edges in enumerate(cycles):
        for e_idx in cy_edges:
            counts[cy_idx][colors[e_idx]] += 1
            
    # Calculate initial energy
    energy = 0
    for cy_idx in range(num_cycles):
        for c in range(4):
            if counts[cy_idx][c] == 6:
                energy += 1
                
    if energy == 0:
        return {'n': n, 'colors': colors}
        
    # Local Search
    T = 1.0
    cooling = 0.995
    
    # Precompute list of violating cycles for faster selection?
    # We can just scan.
    
    while time.time() < deadline:
        # Pick an edge
        if random.random() < 0.7:
            # Find a violating cycle
            # Optimization: Keep a list of violating cycle indices?
            # Updating it is expensive.
            # Just scan a random subset?
            # Or just pick random edge from random cycle?
            # Let's pick a random cycle and check if it's violating.
            # If not, pick random edge.
            
            # Better: Pick random edge, check if it's in a violation.
            # This is O(1) to pick, O(cycles_per_edge) to check.
            # We already have edge_to_cycles.
            
            e_idx = random.randint(0, num_edges - 1)
            # Check if any cycle containing e_idx is violating
            is_in_violation = False
            for cy in edge_to_cycles[e_idx]:
                if any(counts[cy][c] == 6 for c in range(4)):
                    is_in_violation = True
                    break
            
            if not is_in_violation and random.random() < 0.5:
                # Pick a random violating cycle if possible
                # This requires scanning all cycles, which is slow.
                # Let's just stick with the random edge strategy.
                pass
                
        else:
            e_idx = random.randint(0, num_edges - 1)
            
        c_old = colors[e_idx]
        
        # Try all 3 other colors
        best_delta = 100
        best_c_new = c_old
        deltas = []
        
        for c_new in range(4):
            if c_new == c_old:
                continue
            delta = 0
            for cy in edge_to_cycles[e_idx]:
                if counts[cy][c_old] == 6:
                    delta -= 1
                if counts[cy][c_new] == 5:
                    delta += 1
            deltas.append((delta, c_new))
            
        min_delta = min(d[0] for d in deltas)
        best_moves = [d[1] for d in deltas if d[0] == min_delta]
        c_new = random.choice(best_moves)
        delta = min_delta
        
        # Acceptance
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0.01:
                prob = pow(2.718281828, -delta / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            colors[e_idx] = c_new
            for cy in edge_to_cycles[e_idx]:
                counts[cy][c_old] -= 1
                counts[cy][c_new] += 1
            energy += delta
            
            if energy == 0:
                return {'n': n, 'colors': colors}
                
        T *= cooling
        
    return None
