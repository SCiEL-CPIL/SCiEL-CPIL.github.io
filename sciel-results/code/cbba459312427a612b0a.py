import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s kill
    
    n = 15
    total_possible = n * (n - 1) // 2  # 105
    
    def check_valid(adj, deg):
        """Check if graph with adjacency list adj and degrees deg satisfies the constraint.
        For every edge (u,v), the set of degrees of N(u)\{v} and N(v)\{u} must be disjoint."""
        for u in range(n):
            for v in adj[u]:
                if v < u:
                    continue
                # Get degrees of neighbors of u excluding v
                degs_u = set()
                for w in adj[u]:
                    if w != v:
                        degs_u.add(deg[w])
                # Get degrees of neighbors of v excluding u
                degs_v = set()
                for w in adj[v]:
                    if w != u:
                        degs_v.add(deg[w])
                # Check intersection
                if degs_u & degs_v:
                    return False
        return True
    
    def edge_count(adj):
        return sum(len(a) for a in adj) // 2
    
    def try_add_edge(adj, deg, edge_set, u, v):
        """Try to add edge (u,v), return True if valid."""
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        if u == v:
            return False
        
        # Tentatively add
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
        
        # Check validity - only need to check edges involving u or v
        valid = True
        for x in adj[u]:
            if x == v:
                continue
            # Check edge (u, x)
            degs_u = set()
            for w in adj[u]:
                if w != x:
                    degs_u.add(deg[w])
            degs_x = set()
            for w in adj[x]:
                if w != u:
                    degs_x.add(deg[w])
            if degs_u & degs_x:
                valid = False
                break
            # Also check edge (v, x) if it exists
            if x in adj[v]:
                degs_v = set()
                for w in adj[v]:
                    if w != x:
                        degs_v.add(deg[w])
                degs_x2 = set()
                for w in adj[x]:
                    if w != v:
                        degs_x2.add(deg[w])
                if degs_v & degs_x2:
                    valid = False
                    break
        
        if valid:
            for y in adj[v]:
                if y == u:
                    continue
                # Check edge (v, y)
                degs_v = set()
                for w in adj[v]:
                    if w != y:
                        degs_v.add(deg[w])
                degs_y = set()
                for w in adj[y]:
                    if w != v:
                        degs_y.add(deg[w])
                if degs_v & degs_y:
                    valid = False
                    break
        
        if not valid:
            # Remove
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            return False
        return True
    
    def greedy_fill(adj, deg, edge_set, edge_order, deadline):
        """Greedy: try to add edges in the given order if they maintain validity."""
        for (u, v) in edge_order:
            if time.time() > deadline:
                break
            if try_add_edge(adj, deg, edge_set, u, v):
                edge_set.add((u, v))
        return edge_count(adj)
    
    # Start with K_{7,8} bipartite graph as baseline
    best_adj = [[] for _ in range(n)]
    best_deg = [0] * n
    best_edge_set = set()
    
    # Initialize with K_{7,8}
    for i in range(7):
        for j in range(7, 15):
            best_adj[i].append(j)
            best_adj[j].append(i)
            best_deg[i] += 1
            best_deg[j] += 1
            best_edge_set.add((i, j))
    
    best_edges = edge_count(best_adj)
    
    # Try different bipartite splits
    for split in range(3, 13):
        if time.time() > deadline - 30:
            break
        
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edge_set = set()
        
        # Initialize with K_{split, n-split}
        for i in range(split):
            for j in range(split, n):
                adj[i].append(j)
                adj[j].append(i)
                deg[i] += 1
                deg[j] += 1
                edge_set.add((i, j))
        
        # Greedy fill with remaining edges
        all_edges = list(combinations(range(n), 2))
        remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set and (v, u) not in edge_set]
        
        cnt = greedy_fill(adj, deg, edge_set, remaining, deadline - 25)
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = [list(a) for a in adj]
            best_deg = list(deg)
            best_edge_set = set(edge_set)
    
    # Local search: try to add missing edges
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in range(n):
                if time.time() > deadline - 5:
                    break
                for v in range(u + 1, n):
                    if (u, v) in best_edge_set or (v, u) in best_edge_set:
                        continue
                    # Try adding
                    best_adj[u].append(v)
                    best_adj[v].append(u)
                    best_deg[u] += 1
                    best_deg[v] += 1
                    
                    valid = True
                    for x in best_adj[u]:
                        if x == v:
                            continue
                        degs_u = set()
                        for w in best_adj[u]:
                            if w != x:
                                degs_u.add(best_deg[w])
                        degs_x = set()
                        for w in best_adj[x]:
                            if w != u:
                                degs_x.add(best_deg[w])
                        if degs_u & degs_x:
                            valid = False
                            break
                        if x in best_adj[v]:
                            degs_v = set()
                            for w in best_adj[v]:
                                if w != x:
                                    degs_v.add(best_deg[w])
                            degs_x2 = set()
                            for w in best_adj[x]:
                                if w != v:
                                    degs_x2.add(best_deg[w])
                            if degs_v & degs_x2:
                                valid = False
                                break
                    if valid:
                        for y in best_adj[v]:
                            if y == u:
                                continue
                            degs_v = set()
                            for w in best_adj[v]:
                                if w != y:
                                    degs_v.add(best_deg[w])
                            degs_y = set()
                            for w in best_adj[y]:
                                if w != v:
                                    degs_y.add(best_deg[w])
                            if degs_v & degs_y:
                                valid = False
                                break
                    
                    if valid:
                        best_edge_set.add((u, v))
                        best_edges += 1
                        improved = True
                    else:
                        best_adj[u].remove(v)
                        best_adj[v].remove(u)
                        best_deg[u] -= 1
                        best_deg[v] -= 1
    
    # Random restarts with different edge orderings
    rng = random.Random(12345)
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        adj2 = [[] for _ in range(n)]
        deg2 = [0] * n
        edge_set2 = set()
        
        # Random order of edges
        all_edges = list(combinations(range(n), 2))
        rng.shuffle(all_edges)
        
        cnt2 = greedy_fill(adj2, deg2, edge_set2, all_edges, deadline - 5)
        
        if cnt2 > best_edges:
            best_edges = cnt2
            best_adj = [list(a) for a in adj2]
            best_deg = list(deg2)
            best_edge_set = set(edge_set2)
    
    # Build result
    result = []
    for (u, v) in sorted(best_edge_set):
        result.append([int(u), int(v)])
    
    # Final validation
    final_adj = [[] for _ in range(n)]
    final_deg = [0] * n
    for e in result:
        u, v = e
        final_adj[u].append(v)
        final_adj[v].append(u)
        final_deg[u] += 1
        final_deg[v] += 1
    
    valid = True
    for u in range(n):
        for v in final_adj[u]:
            if v < u:
                continue
            degs_u = set()
            for w in final_adj[u]:
                if w != v:
                    degs_u.add(final_deg[w])
            degs_v = set()
            for w in final_adj[v]:
                if w != u:
                    degs_v.add(final_deg[w])
            if degs_u & degs_v:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: star graph
        return [[0, v] for v in range(1, n)]
    
    return result

# EVOLVE-BLOCK-END
