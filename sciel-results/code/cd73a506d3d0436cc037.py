import random
import time
import itertools

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    start_time = time.time()
    timeout = 115  # Safety margin
    
    def has_k33(adj):
        # Check if graph has K3,3 using bitmasks for speed
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if len(common) >= 3:
                        return True, (u1, u2, u3, common)
        return False, None

    def count_violations(adj):
        # Returns list of (u1, u2, u3, common_set)
        violations = []
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if len(common) >= 3:
                        violations.append((u1, u2, u3, common))
        return violations

    def remove_edge(adj, u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    def add_edge(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)

    def is_add_safe(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        for b1 in adj[u]:
            for b2 in adj[u]:
                if b1 == b2: continue
                if b1 == v or b2 == v: continue
                common = adj[v] & adj[b1] & adj[b2]
                common -= {u, v, b1, b2}
                if len(common) >= 2:
                    return False
        return True

    def direct_construct():
        # 5x4 grid construction: partition 20 vertices into 5 groups of 4
        # Each group i connects to groups (i+1)%5 and (i+2)%5
        # This creates a bipartite-like structure that avoids K3,3
        adj = [set() for _ in range(n)]
        groups = [[i*4+j for j in range(4)] for i in range(5)]
        
        # Connect groups in a pattern that avoids K3,3
        # Group i connects to all vertices in groups (i+1)%5 and (i+2)%5
        for i in range(5):
            for j in [(i+1)%5, (i+2)%5]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Verify no K3,3 and count edges
        has_violation, _ = has_k33(adj)
        edge_count = sum(len(s) for s in adj) // 2
        if not has_violation:
            return adj, edge_count
        return None, 0

    def local_search(adj_init, max_swaps=5000):
        # Work on a copy
        adj = [set(s) for s in adj_init]
        
        # Phase 1: Remove edges to fix K3,3
        while True:
            violations = count_violations(adj)
            if not violations:
                break
            
            edge_counts = {}
            for u1, u2, u3, C in violations:
                for c in C:
                    edge_counts[(u1, c)] = edge_counts.get((u1, c), 0) + 1
                    edge_counts[(u2, c)] = edge_counts.get((u2, c), 0) + 1
                    edge_counts[(u3, c)] = edge_counts.get((u3, c), 0) + 1
            
            if not edge_counts:
                break
                
            max_count = -1
            best_edge = None
            for e, cnt in edge_counts.items():
                if cnt > max_count:
                    max_count = cnt
                    best_edge = e
            
            if best_edge:
                remove_edge(adj, best_edge[0], best_edge[1])
        
        # Phase 2: Add edges greedily
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        
        # Phase 3: 1-edge swaps (remove one, add one) to improve edge count
        edge_count = sum(len(s) for s in adj) // 2
        swaps_made = 0
        while swaps_made < max_swaps and time.time() - start_time < timeout:
            made_swap = False
            # Try all possible single edge removals
            current_edges = set()
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        current_edges.add((i, j))
            
            for e in list(current_edges):
                u, v = e
                # Temporarily remove
                remove_edge(adj, u, v)
                
                # Try to add a different edge
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]:
                            if is_add_safe(adj, i, j):
                                add_edge(adj, i, j)
                                # Check if this improves (we removed 1, added 1, so same count)
                                # But maybe it enables more additions later
                                # For now, just keep the swap if it's valid
                                made_swap = True
                                swaps_made += 1
                                break
                    if made_swap:
                        break
                
                if not made_swap:
                    # Revert
                    add_edge(adj, u, v)
                
                if made_swap:
                    break
            
            if not made_swap:
                break
        
        # Final greedy add
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        
        edge_count = sum(len(s) for s in adj) // 2
        return adj, edge_count

    def random_init(p):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(adj, i, j)
        return adj

    # First try direct construction
    adj_direct, edge_count_direct = direct_construct()
    if adj_direct is not None:
        adj_ls, edge_count_ls = local_search(adj_direct)
        if edge_count_ls > best_edges:
            best_edges = edge_count_ls
            edges = []
            for i in range(n):
                for j in adj_ls[i]:
                    if i < j:
                        edges.append([i, j])
            best_graph = edges
    
    # Then try random restarts with local search
    for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for _ in range(15):
            if time.time() - start_time > timeout:
                break
            
            adj_rand = random_init(p)
            adj_ls, edge_count_ls = local_search(adj_rand)
            if edge_count_ls > best_edges:
                best_edges = edge_count_ls
                edges = []
                for i in range(n):
                    for j in adj_ls[i]:
                        if i < j:
                            edges.append([i, j])
                best_graph = edges
        
        if time.time() - start_time > timeout:
            break

    if best_graph is None:
        # Fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
