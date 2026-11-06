import time
import random

def _has_k222_bitset(colors, n):
    """
    Check if the coloring contains a monochromatic K2,2,2 using bitsets.
    """
    adj = [[0] * n for _ in range(2)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                if not (adj[c][u] & (1 << v)):
                    continue
                common = adj[c][u] & adj[c][v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx_v = lsb.bit_length() - 1
                    verts.append(idx_v)
                    mask ^= lsb
                    
                if len(verts) < 4:
                    continue
                    
                for i_idx in range(len(verts)):
                    a = verts[i_idx]
                    n_a = adj[c][a] & common
                    for j_idx in range(i_idx+1, len(verts)):
                        b = verts[j_idx]
                        n_b = adj[c][b] & common
                        common2 = n_a & n_b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        if common2 & (common2 - 1):
                            return True
    return False

def _count_violations_fast(colors, n):
    """
    Fast count of violations using incremental-friendly structure.
    Returns the number of (u,v,c) triples that have a K2,2 in their common neighborhood.
    """
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                if not (adj[c][u] & (1 << v)):
                    continue
                common = adj[c][u] & adj[c][v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Quick check: need at least 4 vertices
                if common.bit_count() < 4:
                    continue
                    
                # Check for K2,2 in common neighborhood
                # For each pair (a,b) in common, check if they share >=2 neighbors in common
                found = False
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx_v = lsb.bit_length() - 1
                    verts.append(idx_v)
                    mask ^= lsb
                    
                for i_idx in range(len(verts)):
                    if found:
                        break
                    a = verts[i_idx]
                    n_a = adj[c][a] & common
                    for j_idx in range(i_idx+1, len(verts)):
                        b = verts[j_idx]
                        n_b = adj[c][b] & common
                        common2 = n_a & n_b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        if common2 & (common2 - 1):
                            count += 1
                            found = True
                            break
    return count

def _count_violations_incremental(colors, n, prev_adj=None):
    """
    Count violations, optionally reusing previous adjacency for speed.
    """
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                if not (adj[c][u] & (1 << v)):
                    continue
                common = adj[c][u] & adj[c][v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                if common.bit_count() < 4:
                    continue
                    
                found = False
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx_v = lsb.bit_length() - 1
                    verts.append(idx_v)
                    mask ^= lsb
                    
                for i_idx in range(len(verts)):
                    if found:
                        break
                    a = verts[i_idx]
                    n_a = adj[c][a] & common
                    for j_idx in range(i_idx+1, len(verts)):
                        b = verts[j_idx]
                        n_b = adj[c][b] & common
                        common2 = n_a & n_b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        if common2 & (common2 - 1):
                            count += 1
                            found = True
                            break
    return count

def _paley_coloring(n):
    """
    Generate Paley coloring for prime n = 4k+1.
    """
    if n % 4 != 1:
        return None
    squares = {x*x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _random_coloring(n):
    """Generate random coloring."""
    num_edges = n * (n - 1) // 2
    return [random.choice([0, 1]) for _ in range(num_edges)]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Strategy: Try to find n=30 coloring using local search
    n_target = 30
    num_edges = n_target * (n_target - 1) // 2
    
    best_cost = float('inf')
    best_colors = None
    
    # Search budget for n=30
    search_deadline = start_time + 120
    
    # Try multiple restarts with different initialization strategies
    for restart in range(200):
        if time.time() > search_deadline:
            break
            
        # Different initialization strategies
        if restart == 0:
            # Start with structured initialization based on Paley-like pattern
            curr = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    # Use a heuristic: color based on distance mod some value
                    dist = (j - i) % 29
                    if dist in {1, 2, 4, 8, 16}:  # Quadratic residues mod 29
                        curr.append(0)
                    else:
                        curr.append(1)
        elif restart == 1:
            # Pure random
            curr = _random_coloring(n_target)
        elif restart == 2:
            # Start from n=29 Paley and extend
            base = _paley_coloring(29)
            if base is not None:
                curr = base[:]
                # Add edges for vertex 29
                for i in range(29):
                    # Color based on heuristic
                    curr.append(random.choice([0, 1]))
            else:
                curr = _random_coloring(n_target)
        elif restart == 3:
            # Another structured init
            curr = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    # Use XOR-based pattern
                    if ((i ^ j) & 1):
                        curr.append(0)
                    else:
                        curr.append(1)
        else:
            # Random restarts
            curr = _random_coloring(n_target)
            
        curr_cost = _count_violations_fast(curr, n_target)
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_colors = curr[:]
            
        if curr_cost == 0:
            return {'n': n_target, 'colors': curr}
            
        # Local Search (Simulated Annealing)
        temp = 1.0
        no_improve_count = 0
        max_no_improve = 3000
        step_limit = 80000
        
        for step in range(step_limit):
            if time.time() > search_deadline:
                break
                
            edge_idx = random.randint(0, num_edges - 1)
            old_c = curr[edge_idx]
            new_c = 1 - old_c
            curr[edge_idx] = new_c
            
            new_cost = _count_violations_fast(curr, n_target)
            
            if new_cost < curr_cost:
                curr_cost = new_cost
                no_improve_count = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = curr[:]
                if new_cost == 0:
                    return {'n': n_target, 'colors': curr}
            else:
                delta = new_cost - curr_cost
                if delta == 0:
                    curr_cost = new_cost
                    no_improve_count += 1
                else:
                    prob = pow(2.71828, -delta / temp)
                    if random.random() > prob:
                        curr[edge_idx] = old_c
                        no_improve_count += 1
                    else:
                        curr_cost = new_cost
                        no_improve_count += 1
                        
            temp *= 0.9995
            if temp < 0.01:
                temp = 0.01
                
            if no_improve_count > max_no_improve:
                break
    
    # 2. Fall back to n=29 Paley coloring
    n_fallback = 29
    colors_29 = _paley_coloring(n_fallback)
    
    if colors_29 is not None:
        if not _has_k222_bitset(colors_29, n_fallback):
            return {'n': n_fallback, 'colors': colors_29}
            
    # 3. Fallback to n=17
    n_fallback2 = 17
    colors_17 = _paley_coloring(n_fallback2)
    if colors_17 is not None and not _has_k222_bitset(colors_17, n_fallback2):
        return {'n': n_fallback2, 'colors': colors_17}
        
    return {'n': 1, 'colors': []}
