import time
import random
import numpy as np

def _count_k222(adj, n):
    """
    Count the number of K2,2,2 subgraphs in the graph defined by adj (color 0).
    Returns the count.
    """
    count = 0
    for y1 in range(n):
        for y2 in range(y1 + 1, n):
            # Common neighbors of y1 and y2 in color 0
            A_mask = adj[y1] & adj[y2]
            # Exclude y1, y2 themselves (though they are not neighbors of themselves)
            A_mask &= ~(1 << y1)
            A_mask &= ~(1 << y2)
            
            # Get list of vertices in A
            A_list = []
            m = A_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                A_list.append(idx)
                m ^= lsb
            
            L = len(A_list)
            if L < 2:
                continue
                
            # For each pair z1, z2 in A, check if they have >= 2 common neighbors in A
            for i in range(L):
                z1 = A_list[i]
                for j in range(i + 1, L):
                    z2 = A_list[j]
                    # Common neighbors of z1 and z2 in color 0, intersected with A
                    common_z = adj[z1] & adj[z2] & A_mask
                    if common_z.bit_count() >= 2:
                        count += 1
    return count

def _find_violation_edges(adj, n, max_edges=20):
    """
    Find up to max_edges edges that are part of a K2,2,2 in either color class.
    Returns list of (u, v) tuples.
    """
    violations = []
    full = (1 << n) - 1
    
    # Helper to check and add violations for a given adjacency list (color 0 or 1)
    def _check_color(adj_list, color_id):
        for y1 in range(n):
            for y2 in range(y1 + 1, n):
                A_mask = adj_list[y1] & adj_list[y2]
                A_mask &= ~(1 << y1)
                A_mask &= ~(1 << y2)
                
                A_list = []
                m = A_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    A_list.append(idx)
                    m ^= lsb
                
                L = len(A_list)
                if L < 2:
                    continue
                    
                for i in range(L):
                    z1 = A_list[i]
                    for j in range(i + 1, L):
                        z2 = A_list[j]
                        common_z = adj_list[z1] & adj_list[z2] & A_mask
                        if common_z.bit_count() >= 2:
                            # We found a K2,2,2 involving y1, y2, z1, z2 and some x1, x2 in common_z
                            # Add edges (y1, y2), (y1, z1), (y1, z2), (y2, z1), (y2, z2), (z1, z2)
                            # Actually, adding (y1, y2) is a good candidate to flip.
                            violations.append((y1, y2))
                            if len(violations) >= max_edges:
                                return
    # Check color 0
    _check_color(adj, 0)
    if len(violations) >= max_edges:
        return violations

    # Check color 1 (complement)
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
        
    _check_color(comp, 1)
    
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Target n=30
    n_target = 30
    q = 29
    
    # Initialize with Paley graph on 29 vertices
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
        
    n = n_target
    adj = [0] * n
    
    # Fill first 29 vertices with Paley graph
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initialize vertex 29 (index 29) randomly
    for i in range(q):
        if random.random() < 0.5:
            adj[i] |= (1 << 29)
            adj[29] |= (1 << i)

    best_adj = None
    best_score = float('inf')
    
    # Simulated Annealing / Local Search
    T = 1.0
    cooling = 0.999
    
    for iteration in range(20000):
        if time.time() > deadline - 5:
            break
            
        # Count violations in both colors
        v0 = _count_k222(adj, n)
        
        # Compute complement for color 1
        full = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full ^ adj[i] ^ (1 << i)
            
        v1 = _count_k222(comp, n)
        
        current_violations = v0 + v1
        
        if current_violations == 0:
            best_adj = adj[:]
            best_score = 0
            break
        
        if current_violations < best_score:
            best_score = current_violations
            best_adj = adj[:]
            
        # Find a violating edge to flip
        violations = _find_violation_edges(adj, n, max_edges=10)
        if not violations:
            # This shouldn't happen if current_violations > 0, but safety check
            break
            
        u, v = random.choice(violations)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Recalculate violations after flip
        v0_new = _count_k222(adj, n)
        
        comp_new = [0] * n
        for i in range(n):
            comp_new[i] = full ^ adj[i] ^ (1 << i)
            
        v1_new = _count_k222(comp_new, n)
        
        new_violations = v0_new + v1_new
        
        delta = new_violations - current_violations
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (random.random() < (2.718281828 ** (-delta / max(T, 0.001)))):
            # Accept move
            pass
        else:
            # Reject move, undo flip
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        T *= cooling
        
        if T < 0.01:
            T = 0.01

    if best_score == 0 and best_adj is not None:
        # Valid coloring found for n=30
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
        
    # Fallback to n=29
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
