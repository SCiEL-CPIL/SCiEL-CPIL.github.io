"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
_DEADLINE = 145.0  # seconds; safety margin before 150s kill

def _paley17():
    """Return adjacency list for Paley graph on 17 vertices."""
    n = 17
    squares = {x * x % n for x in range(1, n)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _count_k4(adj, n):
    """Count K4s in graph using adjacency lists."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common_ij = adj[i] & adj[j]
            for k in common_ij:
                if k <= j:
                    continue
                count += len(adj[k] & adj[i] & adj[j])
    return count

def _count_indep6(adj, n):
    """Count independent sets of size 6 using complement."""
    # Independent set in G = clique in complement
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp[i].add(j)
    # Count K6 in complement
    count = 0
    for i in range(n):
        for j in comp[i]:
            if j <= i:
                continue
            c1 = comp[i] & comp[j]
            for k in c1:
                if k <= j:
                    continue
                c2 = c1 & comp[k]
                for l in c2:
                    if l <= k:
                        continue
                    c3 = c2 & comp[l]
                    for m in c3:
                        if m <= l:
                            continue
                        count += len(c3 & comp[m])
    return count

def _is_valid(adj, n):
    """Check no K4 and no independent 6-set."""
    # Check K4
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common_ij = adj[i] & adj[j]
            for k in common_ij:
                if k <= j:
                    continue
                if adj[k] & adj[i] & adj[j]:
                    return False
    # Check independent 6-set
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp[i].add(j)
    for i in range(n):
        for j in comp[i]:
            if j <= i:
                continue
            c1 = comp[i] & comp[j]
            for k in c1:
                if k <= j:
                    continue
                c2 = c1 & comp[k]
                for l in c2:
                    if l <= k:
                        continue
                    c3 = c2 & comp[l]
                    for m in c3:
                        if m <= l:
                            continue
                        if c3 & comp[m]:
                            return False
    return True

def _try_add_vertex(adj, n, new_v, candidates):
    """Try to add a new vertex connected to 'candidates' subset of existing vertices."""
    new_adj = [set(s) for s in adj]
    new_adj.append(set(candidates))
    for v in candidates:
        new_adj[v].add(n)
    return new_adj

def _search():
    """Local search: start from Paley-17, try to grow by adding vertices."""
    start = time.time()
    adj = _paley17()
    n = 17
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow: add vertices with carefully chosen neighborhoods
    for attempt in range(200):
        if time.time() - start > _DEADLINE:
            break
        if n >= 35:
            break
        
        # Try adding a vertex
        success = False
        for trial in range(50):
            if time.time() - start > _DEADLINE:
                break
            
            # Choose a random subset of existing vertices as neighbors
            # Bias toward moderate density
            prob = 0.4
            candidates = set()
            for v in range(n):
                if random.random() < prob:
                    candidates.add(v)
            
            # Ensure the new vertex doesn't create K4
            # A K4 involving new_v requires 3 mutually adjacent neighbors
            valid_neighborhood = True
            for a in candidates:
                for b in candidates:
                    if a >= b:
                        continue
                    if b in adj[a]:
                        for c in candidates:
                            if c <= b:
                                continue
                            if c in adj[a] and c in adj[b]:
                                valid_neighborhood = False
                                break
                    if not valid_neighborhood:
                        break
                if not valid_neighborhood:
                    break
            if not valid_neighborhood:
                continue
            
            # Check no independent 6-set with new vertex
            # New vertex is non-adjacent to vertices NOT in candidates
            non_neighbors = set(range(n)) - candidates
            # An independent 6-set containing new_v needs 5 mutually non-adjacent vertices from non_neighbors
            # i.e., a K5 in the complement restricted to non_neighbors
            has_indep6_with_new = False
            nn_list = list(non_neighbors)
            if len(nn_list) >= 5:
                # Check if non_neighbors contains an independent set of size 5 in G
                # = clique of size 5 in complement
                # Quick check: if |non_neighbors| < 5, skip
                # Use simple check for small sets
                if len(nn_list) <= 15:
                    # Check all 5-subsets is too slow; use greedy heuristic
                    # Count how many edges are in non_neighbors (in G)
                    # If too few edges, likely has independent 5
                    nn_set = set(nn_list)
                    edges_in_nn = 0
                    for u in nn_list:
                        for v in nn_list:
                            if u < v and v in adj[u]:
                                edges_in_nn += 1
                    # Turan: max edges in K5-free on m vertices
                    # For independent 5 in G, we need complement to have K5
                    # Heuristic: if edges_in_nn is small relative to max possible
                    max_edges = len(nn_list) * (len(nn_list) - 1) // 2
                    if max_edges > 0 and edges_in_nn / max_edges < 0.3:
                        # Likely has independent 5
                        has_indep6_with_new = True
                    else:
                        # More careful check for smaller sets
                        if len(nn_list) <= 12:
                            # Check if there's an independent 5 in G among nn_list
                            # = clique 5 in complement
                            comp_nn = [set() for _ in range(len(nn_list))]
                            idx_map = {v: i for i, v in enumerate(nn_list)}
                            for i, u in enumerate(nn_list):
                                for j, v in enumerate(nn_list):
                                    if i != j and v not in adj[u]:
                                        comp_nn[i].add(j)
                            # Find clique 5 in comp_nn
                            found = False
                            for i in range(len(nn_list)):
                                for j in comp_nn[i]:
                                    if j <= i:
                                        continue
                                    c1 = comp_nn[i] & comp_nn[j]
                                    for k in c1:
                                        if k <= j:
                                            continue
                                        c2 = c1 & comp_nn[k]
                                        for l in c2:
                                            if l <= k:
                                                continue
                                            if c2 & comp_nn[l]:
                                                found = True
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                has_indep6_with_new = True
            
            if has_indep6_with_new:
                continue
            
            # Try the addition
            new_adj = _try_add_vertex(adj, n, n, candidates)
            new_n = n + 1
            if _is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                success = True
                if n > best_n:
                    best_n = n
                    best_adj = [set(s) for s in adj]
                break
        
        if not success:
            # Try edge flips to make room
            flipped = False
            for trial in range(20):
                if time.time() - start > _DEADLINE:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if v in adj[u]:
                    # Remove edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                if _is_valid(adj, n):
                    flipped = True
                    break
                else:
                    # Revert
                    if v in adj[u]:
                        adj[u].discard(v)
                        adj[v].discard(u)
                    else:
                        adj[u].add(v)
                        adj[v].add(u)
            if not flipped:
                # Try a few more random restarts of neighborhood choice
                continue
    
    return best_adj, best_n

def construct():
    start = time.time()
    adj, n = _search()
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
