"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _paley_graph(q):
    """Return (n, edges) for the Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    residues.discard(0)
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _has_k5(edges, n):
    """Check if graph has a K5. Returns True if found."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Find all K4s and check if any vertex connects to all 4
    # First find all K3s (triangles)
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        triangles.append((i, j, k))
    
    # Find all K4s from triangles
    for tri in triangles:
        i, j, k = tri
        common = adj[i] & adj[j] & adj[k]
        for l in common:
            if l > k:
                # Found K4: (i, j, k, l)
                # Check if any vertex connects to all 4
                common4 = adj[i] & adj[j] & adj[k] & adj[l]
                if common4:
                    return True
    return False


def _has_ind5(edges, n):
    """Check if graph has an independent set of size 5. Returns True if found."""
    # Independent set of size 5 in G = clique of size 5 in complement
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find K5 in complement
    for i in range(n):
        for j in comp_adj[i]:
            if j > i:
                common = comp_adj[i] & comp_adj[j]
                for k in common:
                    if k > j:
                        common2 = comp_adj[i] & comp_adj[j] & comp_adj[k]
                        for l in common2:
                            if l > k:
                                common3 = comp_adj[i] & comp_adj[j] & comp_adj[k] & comp_adj[l]
                                if common3:
                                    return True
    return False


def _is_valid(edges, n):
    """Check if graph is (5,5)-free."""
    if n < 5:
        return True
    if _has_k5(edges, n):
        return False
    if _has_ind5(edges, n):
        return False
    return True


def _greedy_extend(base_edges, base_n, target_n, seed=42):
    """
    Try to extend a (5,5)-free graph on base_n vertices to target_n vertices
    by greedily adding vertices with carefully chosen edges.
    Returns (n, edges) or None if fails.
    """
    rng = random.Random(seed)
    edges = list(base_edges)
    n = base_n
    
    # Build adjacency from base
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # For each new vertex, try to add it with a random subset of edges
    # that doesn't create a K5 or independent 5-set
    
    for new_v in range(base_n, target_n):
        # Try multiple random edge assignments
        found = False
        for attempt in range(200):
            # Randomly decide which existing vertices to connect to
            # Use a biased strategy: connect to ~half the vertices
            new_adj = set()
            for v in range(n):
                if rng.random() < 0.5:
                    new_adj.add(v)
            
            # Check validity: adding vertex new_v with adjacency new_adj
            # K5 violation: there exists a K4 in the subgraph on vertices connected to new_v
            #   such that new_v connects to all 4
            # Ind5 violation: there exists an independent set of size 4 in the subgraph
            #   on vertices NOT connected to new_v, and new_v is not connected to any of them
            
            # Efficient check:
            # For K5: find if any K4 exists among new_adj vertices
            # For Ind5: find if any independent 4-set exists among non-new_adj vertices
            
            # K4 check among new_adj
            k4_found = False
            new_adj_list = list(new_adj)
            for i_idx in range(len(new_adj_list)):
                a = new_adj_list[i_idx]
                for j_idx in range(i_idx + 1, len(new_adj_list)):
                    b = new_adj_list[j_idx]
                    if b not in adj[a]:
                        continue
                    common_ab = adj[a] & adj[b]
                    for k_idx in range(j_idx + 1, len(new_adj_list)):
                        c = new_adj_list[k_idx]
                        if c not in common_ab:
                            continue
                        common_abc = common_ab & adj[c]
                        for d in common_abc:
                            if d in new_adj:
                                k4_found = True
                                break
                        if k4_found:
                            break
                    if k4_found:
                        break
                if k4_found:
                    break
            if k4_found:
                continue
            
            # Independent 4-set check among non-new_adj
            non_adj = set(range(n)) - new_adj
            ind4_found = False
            non_adj_list = list(non_adj)
            for i_idx in range(len(non_adj_list)):
                a = non_adj_list[i_idx]
                for j_idx in range(i_idx + 1, len(non_adj_list)):
                    b = non_adj_list[j_idx]
                    if b in adj[a]:
                        continue
                    non_common_ab = (set(range(n)) - adj[a]) & (set(range(n)) - adj[b])
                    for k_idx in range(j_idx + 1, len(non_adj_list)):
                        c = non_adj_list[k_idx]
                        if c not in non_common_ab:
                            continue
                        non_common_abc = non_common_ab & (set(range(n)) - adj[c])
                        for d in non_common_abc:
                            if d in non_adj:
                                ind4_found = True
                                break
                        if ind4_found:
                            break
                    if ind4_found:
                        break
                if ind4_found:
                    break
            if ind4_found:
                continue
            
            # Valid! Add the vertex
            for v in new_adj:
                edges.append((min(v, new_v), max(v, new_v)))
                adj[v].add(new_v)
            adj.append(set(new_adj))
            n += 1
            found = True
            break
        
        if not found:
            return None
    
    return n, edges


def construct():
    """
    Construct a (5,5)-free graph. Start from Paley(37) and try to extend it.
    Returns (n, edges).
    """
    deadline = time.time() + 270  # 300s limit, keep 30s margin
    
    # Start with Paley(37)
    base_n, base_edges = _paley_graph(37)
    
    # Verify base is valid
    if not _is_valid(base_edges, base_n):
        # Shouldn't happen, but fallback
        return base_n, base_edges
    
    best_n = base_n
    best_edges = base_edges
    
    # Try to extend with different seeds and strategies
    # First, try a few quick greedy extensions
    for seed in range(100):
        if time.time() > deadline - 60:
            break
        result = _greedy_extend(base_edges, base_n, base_n + 1, seed=seed)
        if result is not None:
            n, edges = result
            if _is_valid(edges, n):
                if n > best_n:
                    best_n = n
                    best_edges = edges
    
    # Try to extend further from the best found so far
    for target_offset in range(1, 6):
        if time.time() > deadline - 30:
            break
        target_n = best_n + target_offset
        for seed in range(50):
            if time.time() > deadline - 30:
                break
            result = _greedy_extend(best_edges, best_n, target_n, seed=seed)
            if result is not None:
                n, edges = result
                if _is_valid(edges, n):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
