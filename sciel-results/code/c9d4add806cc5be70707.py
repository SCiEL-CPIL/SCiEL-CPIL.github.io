import time
import random
import sys

def _build_k32():
    """Build the 32-vertex 4-coloring from GF(16) cubic residues."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

_K32 = _build_k32()

def _edge_idx(i, j, n):
    """Index of edge (i,j) in the flat colors array (i<j)."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _validate_fast(colors, n):
    """Fast validation: no color class contains a triangle."""
    # Build adjacency bitmasks for each color
    adj = [0] * (4 * n)  # adj[c*n + v] = bitmask of neighbors of v in color c
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1

    for c in range(4):
        base = c * n
        for i in range(n):
            nbrs = adj[base + i]
            # For each neighbor j of i in color c, check if any neighbor of j is also a neighbor of i
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                # Check if j's neighbors in color c intersect with i's neighbors
                if adj[base + j] & nbrs:
                    return False
    return True

def _add_vertex(colors, n, deadline):
    """
    Try to add one vertex to a valid n-vertex coloring.
    Returns (new_colors, n+1) on success, (None, n) on failure.
    
    Strategy: For each color c, the set S_c of vertices connected to new vertex
    by color c must be an independent set in the color-c graph.
    We use a greedy approach with randomization.
    """
    # Precompute: for each color c, build adjacency lists
    # adj_c[v] = list of neighbors of v in color c
    adj = [[] for _ in range(4)]
    for c in range(4):
        adj[c] = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
            idx += 1

    # For each color c, we need to find an independent set.
    # Greedy: for each vertex, compute for each color c whether it can be added
    # to the current independent set for color c.
    
    best_assignment = None
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Randomly shuffle vertex order for greedy
        order = list(range(n))
        random.shuffle(order)
        
        # assignment[v] = color assigned to edge (v, new_vertex)
        assignment = [-1] * n
        # current_sets[c] = set of vertices already assigned to color c
        current_sets = [set() for _ in range(4)]
        # For quick conflict checking: conflict[c][v] = whether v conflicts with current_sets[c]
        # i.e., v has a neighbor in current_sets[c] via color c
        conflict = [set() for _ in range(4)]
        
        success = True
        for v in order:
            # Find colors where v can be added
            valid_colors = []
            for c in range(4):
                if v not in conflict[c]:
                    valid_colors.append(c)
            
            if not valid_colors:
                success = False
                break
            
            # Choose color: prefer the one with fewest current vertices (balance)
            # or randomly among valid
            if len(valid_colors) == 1:
                c = valid_colors[0]
            else:
                # Weight by balance: prefer colors with fewer vertices
                weights = [1.0 / (len(current_sets[c]) + 1) for c in valid_colors]
                total = sum(weights)
                r = random.random() * total
                cum = 0
                c = valid_colors[0]
                for idx2, cc in enumerate(valid_colors):
                    cum += weights[idx2]
                    if r <= cum:
                        c = cc
                        break
            
            assignment[v] = c
            current_sets[c].add(v)
            
            # Update conflict: for each neighbor u of v in color c, u now conflicts with color c
            for u in adj[c][v]:
                if assignment[u] == -1:  # Only update unassigned vertices
                    conflict[c].add(u)
        
        if success:
            best_assignment = assignment
            break
    
    if best_assignment is None:
        return None, n
    
    # Build new colors
    new_colors = list(colors)
    for v in range(n):
        new_colors.append(best_assignment[v])
    
    return new_colors, n + 1

def _add_vertex_backtrack(colors, n, deadline, max_depth=20):
    """
    Backtracking search to add a vertex. More thorough than greedy.
    """
    # Precompute adjacency for each color
    adj = [[] for _ in range(4)]
    for c in range(4):
        adj[c] = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
            idx += 1
    
    # Order vertices by degree in the "conflict" sense: vertices with high total degree first
    # This makes the search more constrained early
    degree = [0] * n
    for c in range(4):
        for v in range(n):
            degree[v] += len(adj[c][v])
    order = sorted(range(n), key=lambda v: -degree[v])
    
    # Backtracking
    assignment = [-1] * n
    # conflict[c][v] = True if v cannot be assigned color c
    conflict = [set() for _ in range(4)]
    
    # Precompute: for each vertex v and color c, the set of neighbors in color c
    # This is adj[c][v]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        v = order[pos]
        
        # Try colors in order of increasing constraint
        for c in range(4):
            if v in conflict[c]:
                continue
            
            # Assign v to color c
            assignment[v] = c
            
            # Check if this assignment conflicts with any already-assigned vertex
            # Actually, conflict[c] already tracks this, so if v not in conflict[c], it's fine
            # But we need to update conflict for future vertices
            # For each unassigned vertex u that is a neighbor of v in color c,
            # u now conflicts with color c
            newly_conflicted = []
            for u in adj[c][v]:
                if assignment[u] == -1 and u not in conflict[c]:
                    conflict[c].add(u)
                    newly_conflicted.append(u)
            
            if backtrack(pos + 1):
                return True
            
            # Undo
            for u in newly_conflicted:
                conflict[c].discard(u)
            assignment[v] = -1
        
        return False
    
    if backtrack(0):
        new_colors = list(colors)
        for v in range(n):
            new_colors.append(assignment[v])
        return new_colors, n + 1
    
    return None, n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline, 35s margin before 150s kill
    
    best_n = 32
    best_colors = list(_K32)
    
    # Try to extend using greedy first (fast), then backtrack if time permits
    current_n = best_n
    current_colors = best_colors
    
    # Phase 1: Greedy extension
    for _ in range(20):  # Try up to 20 more vertices
        if time.time() > deadline - 30:  # Leave 30s for backtrack phase
            break
        new_colors, new_n = _add_vertex(current_colors, current_n, deadline)
        if new_colors is None:
            break
        current_n = new_n
        current_colors = new_colors
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors
    
    # Phase 2: Try backtrack for one more vertex if greedy got stuck
    if time.time() < deadline - 10:
        new_colors, new_n = _add_vertex_backtrack(current_colors, current_n, deadline - 5)
        if new_colors is not None and new_n > current_n:
            current_n = new_n
            current_colors = new_colors
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors
    
    # Final validation
    if time.time() < deadline:
        if not _validate_fast(best_colors, best_n):
            best_n = 32
            best_colors = list(_K32)
    
    return {'n': best_n, 'colors': best_colors}
