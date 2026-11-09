"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return a nonempty subset of [1, 512] with lcm(a,b)<=512 for every pair.
    Maximize cardinality.
    
    Strategy:
    1. Build compatibility graph where edge (a,b) exists if lcm(a,b) <= 512.
    2. Find maximum clique using Tomita's algorithm with greedy coloring bound.
    3. Use multiple vertex orderings for robustness.
    4. Fallback to greedy min-degree independent set if time runs out.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 512
    numbers = list(range(1, N + 1))
    n = N
    
    # Build compatibility graph (edge if lcm(a,b) <= N)
    # Using bitmasks for fast intersection
    compat = [0] * (n + 1)  # 1-indexed
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = math.gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << j)
        compat[i] = mask
    
    # Convert to 0-indexed bitmasks
    # new_adj[v] where v in 0..n-1 corresponds to number v+1
    new_adj = [0] * n
    for i in range(n):
        # Build mask for node i (number i+1)
        old_mask = compat[i + 1]
        # Convert from 1-indexed to 0-indexed
        mask = 0
        for j in range(n):
            if old_mask & (1 << (j + 1)):
                mask |= (1 << j)
        new_adj[i] = mask
    
    # Degrees
    degrees = [bin(new_adj[i]).count('1') for i in range(n)]
    
    # Best solution found so far
    best_size = 0
    best_clique = []
    
    # Tomita's maximum clique algorithm with greedy coloring
    def expand(R, P, colors, max_color):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
        
        # Prune: if we can't beat best, return
        if len(R) + max_color <= best_size:
            return
        
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Greedy coloring of P
        colors = [0] * n
        max_color = 0
        U = P
        while U:
            max_color += 1
            V = U
            while V:
                # Pick lowest set bit
                v = (V & -V).bit_length() - 1
                colors[v] = max_color
                V &= ~(1 << v)
                U &= ~(1 << v)
                V &= ~new_adj[v]
        
        # Bound check with coloring
        if len(R) + max_color <= best_size:
            return
        
        # Branch: pick vertex with highest color
        v_best = -1
        v_max_color = 0
        P_temp = P
        while P_temp:
            v = (P_temp & -P_temp).bit_length() - 1
            if colors[v] > v_max_color:
                v_max_color = colors[v]
                v_best = v
            P_temp &= ~(1 << v)
        
        if v_best == -1:
            return
        
        # Include v_best
        R.append(v_best)
        P_new = P & new_adj[v_best]
        expand(R, P_new, colors, max_color)
        R.pop()
        
        # Exclude v_best
        P &= ~(1 << v_best)
        expand(R, P, colors, max_color)
    
    # Try multiple orderings
    orderings = [
        list(range(n)),  # natural order
        sorted(range(n), key=lambda x: degrees[x], reverse=True),  # degree descending
        sorted(range(n), key=lambda x: degrees[x]),  # degree ascending
        sorted(range(n), key=lambda x: (degrees[x], x)),  # degree then index
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        # Remap: new index i corresponds to old vertex ordering[i]
        old_to_new = {old: new for new, old in enumerate(ordering)}
        
        # Build remapped adjacency
        remapped_adj = [0] * n
        for new_i in range(n):
            old_i = ordering[new_i]
            mask = 0
            for new_j in range(n):
                old_j = ordering[new_j]
                if new_i == new_j:
                    continue
                if new_adj[old_i] & (1 << old_j):
                    mask |= (1 << new_j)
            remapped_adj[new_i] = mask
        
        # Save and swap
        old_adj = new_adj
        new_adj = remapped_adj
        
        # Run clique search
        initial_P = (1 << n) - 1
        expand([], initial_P, [0] * n, 0)
        
        # Restore
        new_adj = old_adj
    
    # Convert best_clique back to numbers
    if best_clique:
        result = [v + 1 for v in best_clique]
    else:
        # Fallback: greedy min-degree independent set
        result = _greedy_fallback()
    
    # Verify and return
    if len(result) > 1:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a = result[i]
                b = result[j]
                g = math.gcd(a, b)
                if a * b // g > N:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return _greedy_fallback()
    
    return result

def _greedy_fallback():
    """Fallback greedy min-degree independent set."""
    N = 512
    n = N
    numbers = list(range(1, n + 1))
    
    # Build conflict graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = math.gcd(a, b)
            if a * b // g > N:
                adj[i].add(j)
                adj[j].add(i)
    
    remaining = set(range(n))
    solution = set()
    
    while remaining:
        min_deg = float('inf')
        best_node = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = v
                if deg == 0:
                    break
        
        solution.add(best_node)
        neighbors = adj[best_node] & remaining
        remaining.discard(best_node)
        for nb in neighbors:
            remaining.discard(nb)
    
    return [v + 1 for v in solution]
# EVOLVE-BLOCK-END
