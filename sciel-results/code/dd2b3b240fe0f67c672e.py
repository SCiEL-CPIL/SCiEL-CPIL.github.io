"""Search-based construction for R(C4,K11) > n."""
import time
import random

# EVOLVE-BLOCK-START

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(adj, n, limit=11):
    """Find an independent set of size >= limit using greedy + backtracking.
    Returns True if an independent set of size >= limit exists."""
    # Greedy coloring approach: if we can color with <= limit-1 colors, alpha < limit
    # But we want to check if alpha >= limit, so try to find independent set of size limit
    # Use a simple branch and bound
    remaining = set(range(n))
    
    def _branch(rem, current_size):
        if current_size >= limit:
            return True
        if not rem:
            return False
        # Upper bound: current_size + len(rem) < limit
        if current_size + len(rem) < limit:
            return False
        # Pick a vertex with max degree in remaining (to prune)
        best_v = -1
        best_deg = -1
        for v in rem:
            deg_in_rem = len(adj[v] & rem)
            if deg_in_rem > best_deg:
                best_deg = deg_in_rem
                best_v = v
        # Branch 1: include best_v
        rem_without_neighbours = rem - adj[best_v] - {best_v}
        if _branch(rem_without_neighbours, current_size + 1):
            return True
        # Branch 2: exclude best_v
        if _branch(rem - {best_v}, current_size):
            return True
        return False
    
    return _branch(remaining, 0)

def _greedy_alpha(adj, n):
    """Greedy upper bound on independence number."""
    total = 0.0
    for v in range(n):
        total += 1.0 / (len(adj[v]) + 1.0)
    return total

def _can_add_vertex(adj, n, cand_set):
    """Check if adding a new vertex with neighbourhood cand_set keeps C4-free.
    
    Conditions:
    1. For every existing vertex u: |cand_set ∩ N(u)| <= 1
    2. For every pair u,v in cand_set: N(u) ∩ N(v) = empty (no common neighbours in old graph)
    """
    cand_list = list(cand_set)
    # Condition 1: for each existing vertex u, at most 1 neighbour in cand_set
    for u in range(n):
        count = 0
        for v in cand_list:
            if v in adj[u]:
                count += 1
                if count >= 2:
                    return False
    # Condition 2: for each pair in cand_set, no common neighbours in old graph
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            if adj[cand_list[i]] & adj[cand_list[j]]:
                return False
    return True

def _construct_polarity_graph(q):
    """Construct the polarity graph of the projective plane of order q.
    Vertices are points of PG(2,q), edges based on a polarity.
    This graph is C4-free.
    
    For q a prime power, the projective plane has q^2+q+1 points.
    Using the standard polarity (x,y,z) -> (x, -y, z) or similar.
    """
    # Represent points as (x, y, z) with not all zero, up to scaling
    # Use normalized representation: first non-zero coordinate is 1
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coord = 1
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    # Polarity: point p=(a,b,c) is adjacent to point q=(a',b',c') if a*a' + b*b' + c*c' == 0 mod q
    # But we need to be careful: this gives a symmetric relation.
    # Actually for a proper polarity, we use a symmetric bilinear form.
    # Use: a*a' + b*b' + c*c' = 0 (mod q)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            a1, b1, c1 = points[i]
            a2, b2, c2 = points[j]
            if (a1*a2 + b1*b2 + c1*c2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n_pts, 'edges': edges, 'adj': adj}

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Strategy 1: Try polarity graphs for various q
    for q in [3, 4, 5, 7, 9, 11, 13]:
        if time.time() > deadline - 10:
            break
        try:
            result = _construct_polarity_graph(q)
            n = result['n']
            adj = result['adj']
            # Verify C4-free
            if not _verify_c4_free(n, adj):
                continue
            # Check alpha < 11
            has_large_indep = _max_independent_set(adj, n, limit=11)
            if not has_large_indep:
                if n > best_n:
                    best_n = n
                    best_edges = result['edges']
        except Exception:
            continue
    
    # Strategy 2: Greedy construction from scratch
    if time.time() < deadline - 5:
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Build graph greedily
            n = 0
            adj = []
            for v in range(43):
                if time.time() > deadline - 15:
                    break
                # Try to add vertex v with a random neighbourhood
                found = False
                for trial in range(50):
                    # Choose a random subset of existing vertices as neighbours
                    # Size should be moderate
                    if n == 0:
                        cand = set()
                    else:
                        # Random subset of size ~ n/4 to n/3
                        target_size = max(1, n // 4 + random.randint(0, n // 6))
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        cand = set(candidates[:min(target_size, n)])
                    
                    if _can_add_vertex(adj, n, cand):
                        # Check alpha won't exceed 10
                        # Quick check: Caro-Wei bound
                        new_n = n + 1
                        cw = 0.0
                        for u in range(n):
                            new_deg = len(adj[u]) + (1 if u in cand else 0)
                            cw += 1.0 / (new_deg + 1.0)
                        cw += 1.0 / (len(cand) + 1.0)
                        if cw <= 10.5:  # Allow some slack
                            found = True
                            break
                        else:
                            # Try smaller neighbourhood
                            if len(cand) > 2:
                                # Remove some vertices from cand
                                cand_list = list(cand)
                                random.shuffle(cand_list)
                                cand = set(cand_list[:len(cand_list)//2])
                                if _can_add_vertex(adj, n, cand):
                                    cw2 = 0.0
                                    for u in range(n):
                                        new_deg = len(adj[u]) + (1 if u in cand else 0)
                                        cw2 += 1.0 / (new_deg + 1.0)
                                    cw2 += 1.0 / (len(cand) + 1.0)
                                    if cw2 <= 10.5:
                                        found = True
                                        break
                
                if not found:
                    break
                
                # Add vertex
                adj.append(set())
                for u in cand:
                    adj[v].add(u)
                    adj[u].add(v)
                n = v + 1
            
            if n > best_n:
                # Verify
                if _verify_c4_free(n, adj):
                    if not _max_independent_set(adj, n, limit=11):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        best_n = n
                        best_edges = edges
    
    # Strategy 3: If we haven't done well, use the triangle construction as fallback
    if best_n < 30:
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        best_n = 30
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
