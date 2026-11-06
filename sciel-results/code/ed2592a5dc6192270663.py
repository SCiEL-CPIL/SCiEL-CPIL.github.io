"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin for overhead/return
    
    N = 256
    
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Use bitmask for speed
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks (conflict graph)
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Complement graph: edge exists if NOT conflicting
    comp_adj = [0] * M
    for i in range(M):
        comp_adj[i] = ((1 << M) - 1) ^ (1 << i) ^ adj[i]
    
    # Compute degrees in complement graph
    comp_degrees = [bin(comp_adj[i]).count('1') for i in range(M)]
    
    # Maximum Clique via branch-and-bound with coloring
    best_clique = [0] * M
    best_size = [0]
    
    def max_clique_branch_and_bound(order, n):
        """Find maximum clique using branch-and-bound with greedy coloring bound."""
        if n == 0 or time.time() > deadline:
            return
        
        # Greedy coloring to get upper bound
        # Process vertices in the given order
        colors = [0] * n
        color_class_sizes = [0] * (n + 1)
        max_color = 0
        
        for i in range(n):
            v = order[i]
            # Find a color for v that doesn't conflict with already-colored neighbors in this subproblem
            used_colors = [False] * (max_color + 2)
            for j in range(i):
                u = order[j]
                # Check if u and v are adjacent in complement (i.e., can be in same clique)
                if (comp_adj[v] >> u) & 1:
                    used_colors[colors[j]] = True
            
            # Find smallest available color
            c = 0
            while c < len(used_colors) and used_colors[c]:
                c += 1
            if c >= len(used_colors):
                used_colors.append(False)
            used_colors[c] = True
            colors[i] = c
            color_class_sizes[c] += 1
            max_color = max(max_color, c)
        
        # Upper bound is max_color + 1
        if max_color + 1 <= best_size[0]:
            return
        
        # Branch on vertices
        # Try vertices in order of decreasing color (more promising first)
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = order[i]
            c = colors[i]
            
            # Upper bound: current best + remaining vertices with color >= c
            # Simplified: if c + 1 <= best_size[0], prune
            if c + 1 <= best_size[0]:
                continue
            
            # Try adding v to clique
            best_clique[v] = 1
            
            # Build subproblem: neighbors of v in complement, from positions < i
            sub_order = []
            for j in range(i):
                u = order[j]
                if (comp_adj[v] >> u) & 1:
                    sub_order.append(u)
            
            sub_n = len(sub_order)
            max_clique_branch_and_bound(sub_order, sub_n)
            
            best_clique[v] = 0
            
            # Update best if improved
            # (handled in recursive call)
    
    # Better approach: use a standard MCS-style algorithm
    def mcs_max_clique():
        """Maximum clique using branch and bound with coloring."""
        nonlocal best_size, best_clique
        
        # Order vertices by degree (descending) for better coloring
        vertices = list(range(M))
        vertices.sort(key=lambda x: comp_degrees[x], reverse=True)
        
        # Map original vertex to position in ordering
        pos_in_order = [0] * M
        for i, v in enumerate(vertices):
            pos_in_order[v] = i
        
        best_size[0] = 0
        best_clique = [0] * M
        
        def color_bound(order, n):
            """Greedy coloring to compute upper bound and color assignment."""
            colors = [0] * n
            max_color = 0
            for i in range(n):
                v = order[i]
                # Check which colors are used by colored neighbors
                forbidden = [False] * (max_color + 2)
                for j in range(i):
                    u = order[j]
                    if (comp_adj[v] >> u) & 1:
                        forbidden[colors[j]] = True
                c = 0
                while c < len(forbidden) and forbidden[c]:
                    c += 1
                colors[i] = c
                max_color = max(max_color, c)
            return colors, max_color + 1
        
        def expand(order, n, current_size, current_clique):
            """Branch and bound expansion."""
            if n == 0:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    best_clique = current_clique[:]
                return
            
            if time.time() > deadline:
                return
            
            # Color bound
            colors, ub = color_bound(order, n)
            if current_size + ub <= best_size[0]:
                return
            
            # Branch in reverse order (higher colors first)
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    return
                
                v = order[i]
                c = colors[i]
                
                # Prune if even taking all remaining with color <= c can't beat best
                if current_size + c + 1 <= best_size[0]:
                    continue
                
                # Add v to clique
                current_clique[v] = 1
                new_current_size = current_size + 1
                
                # Build sub-order: neighbors of v from positions < i
                sub_order = []
                for j in range(i):
                    u = order[j]
                    if (comp_adj[v] >> u) & 1:
                        sub_order.append(u)
                
                expand(sub_order, len(sub_order), new_current_size, current_clique)
                
                current_clique[v] = 0
        
        # Initial ordering
        expand(vertices, M, 0, [0] * M)
    
    # Run MCS
    mcs_max_clique()
    
    mcs_result = [idx_to_num[i] for i in range(M) if best_clique[i]]
    
    # Also run simulated annealing as fallback/improvement
    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected
    
    degree_order = sorted(range(M), key=lambda i: bin(adj[i]).count('1'))
    sa_best_mask = greedy_init(degree_order)
    sa_best_size = bin(sa_best_mask).count('1')
    
    def simulated_annealing(start_mask, start_size, steps, t_start, t_end):
        current_mask = start_mask
        current_size = start_size
        t = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            v = random.randint(0, M - 1)
            
            if (current_mask >> v) & 1:
                trial_mask = current_mask & ~(1 << v)
                trial_size = current_size - 1
                delta_e = 1
            else:
                neighbors_in_set = adj[v] & current_mask
                if neighbors_in_set == 0:
                    trial_mask = current_mask | (1 << v)
                    trial_size = current_size + 1
                    delta_e = -1
                else:
                    trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                    trial_size = current_size - bin(neighbors_in_set).count('1') + 1
                    delta_e = bin(neighbors_in_set).count('1') - 1
            
            if delta_e <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                if random.random() < math.exp(-delta_e / t):
                    current_mask = trial_mask
                    current_size = trial_size
            
            t *= cooling_rate
        
        return current_mask, current_size
    
    random.seed(42)
    
    # SA from MCS result if it's good, otherwise from greedy
    mcs_mask = 0
    for i in range(M):
        if best_clique[i]:
            mcs_mask |= (1 << i)
    
    if best_size[0] > sa_best_size:
        sa_best_mask = mcs_mask
        sa_best_size = best_size[0]
    
    # Polish with SA
    if time.time() < deadline - 3:
        mask, size = simulated_annealing(sa_best_mask, sa_best_size, 30000, 20.0, 0.05)
        if size > sa_best_size:
            sa_best_mask = mask
            sa_best_size = size
    
    # Pick the best result
    if sa_best_size >= best_size[0]:
        result = [idx_to_num[i] for i in range(M) if (sa_best_mask >> i) & 1]
    else:
        result = mcs_result
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
