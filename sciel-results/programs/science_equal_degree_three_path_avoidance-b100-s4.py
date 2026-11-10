"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin (90s limit, leave buffer)
    
    # Use complete bipartite graph K_{7,8} as base
    # Partition A: {0,1,2,3,4,5,6} (7 vertices)
    # Partition B: {7,8,9,10,11,12,13,14} (8 vertices)
    # In K_{7,8}, every vertex in A has degree 8, every vertex in B has degree 7
    
    A = list(range(7))
    B = list(range(7, 15))
    
    # Base edges: complete bipartite
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    
    def build_edges(extra_edges):
        """Build edge set from base plus extra internal edges."""
        edges = set(base_set)
        for e in extra_edges:
            edges.add(tuple(sorted(e)))
        return edges
    
    def get_degrees(edge_set, n):
        deg = [0] * n
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        """Check if no simple three-edge path has endpoints of equal degree."""
        deg = get_degrees(edge_set, n)
        
        # Build adjacency for efficiency
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v), check all paths s-u-v-t with s != t
        for (u, v) in edge_set:
            # s is a neighbor of u (not v), t is a neighbor of v (not u)
            # and s != t
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    # Path s-u-v-t exists
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with just the base bipartite graph
    best_edges = set(base_set)
    best_count = len(best_edges)
    
    # Generate all possible internal edges (within A and within B)
    internal_candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            internal_candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            internal_candidates.append((B[i], B[j]))
    
    # Greedy approach: try adding edges one by one
    # Sort by some heuristic - edges within same partition
    # Try all single edges first
    valid_single = []
    for e in internal_candidates:
        if time.time() > deadline:
            break
        es = build_edges([e])
        if is_valid(es):
            valid_single.append(e)
    
    # Greedy: keep adding edges that maintain validity
    current_edges = []
    for e in valid_single:
        if time.time() > deadline:
            break
        test_edges = current_edges + [e]
        es = build_edges(test_edges)
        if is_valid(es):
            current_edges.append(e)
    
    if len(current_edges) + len(base_set) > best_count:
        best_edges = build_edges(current_edges)
        best_count = len(best_edges)
    
    # Try to improve with local search: try adding more edges
    # Also try different orderings
    # Try all pairs that weren't in valid_single but might work together
    remaining = [e for e in internal_candidates if e not in current_edges]
    
    # Try adding pairs from remaining
    for i in range(len(remaining)):
        if time.time() > deadline:
            break
        e1 = remaining[i]
        for j in range(i+1, len(remaining)):
            if time.time() > deadline:
                break
            e2 = remaining[j]
            test_edges = current_edges + [e1, e2]
            es = build_edges(test_edges)
            if len(es) <= best_count:
                continue
            if is_valid(es):
                # Check if we can add even more
                current_edges = test_edges
                best_edges = es
                best_count = len(best_edges)
    
    # Try to add more single edges from remaining
    for e in remaining:
        if time.time() > deadline:
            break
        if e in current_edges:
            continue
        test_edges = current_edges + [e]
        es = build_edges(test_edges)
        if len(es) <= best_count:
            continue
        if is_valid(es):
            current_edges = test_edges
            best_edges = es
            best_count = len(best_edges)
    
    # Try a different strategy: start fresh and use a different greedy order
    # Sort internal candidates by "centrality" or some other measure
    # For now, try reversing the order
    valid_single_rev = list(reversed(valid_single))
    current_edges_rev = []
    for e in valid_single_rev:
        if time.time() > deadline:
            break
        test_edges = current_edges_rev + [e]
        es = build_edges(test_edges)
        if is_valid(es):
            current_edges_rev.append(e)
    
    if len(current_edges_rev) + len(base_set) > best_count:
        best_edges = build_edges(current_edges_rev)
        best_count = len(best_edges)
    
    # Convert to list format
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are correct: integers must be Python int
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
# EVOLVE-BLOCK-END
