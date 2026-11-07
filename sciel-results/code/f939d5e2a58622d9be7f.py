import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
    n = 15
    total_possible = n * (n - 1) // 2  # 105
    
    def is_valid(adj, deg):
        """Check if graph with adjacency list adj and degrees deg satisfies the constraint.
        For every edge (u,v), all other neighbors of u must have distinct degrees,
        and all other neighbors of v must have distinct degrees."""
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    s = neighbors[i]
                    t = neighbors[j]
                    # s-u-t is a 2-edge path, not a 3-edge path
                    # We need s-u-v-t where s-u, u-v, v-t are edges and s!=t
                    # Actually the constraint is: for path s-u-v-t (3 edges, 4 vertices),
                    # s and t must have different degrees.
                    # This means: for any edge (u,v), and any s in N(u)\{v}, any t in N(v)\{u},
                    # we need deg[s] != deg[t].
                    # Equivalently: for edge (u,v), the set of degrees of N(u)\{v} and
                    # N(v)\{u} must be disjoint.
                    pass
        
        # More efficient check:
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
    
    # Start with empty graph and greedily add edges
    adj = [[] for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    
    def try_add_edge(u, v):
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
    
    # Greedy construction: try to add all possible edges
    # Use a priority: try edges that are "safe" first
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline - 10:
                break
            if try_add_edge(u, v):
                edge_set.add((u, v))
    
    current_edges = edge_count(adj)
    
    # Local search: try to add more edges by swapping
    # Or try random restarts with different edge orderings
    best_edges = current_edges
    best_adj = [list(a) for a in adj]
    best_deg = list(deg)
    best_edge_set = set(edge_set)
    
    rng = random.Random(12345)
    
    # Multiple restarts with random edge ordering
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        adj2 = [[] for _ in range(n)]
        deg2 = [0] * n
        edge_set2 = set()
        
        # Random order of edges
        all_edges = list(combinations(range(n), 2))
        rng.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 5:
                break
            if u == v:
                continue
            if (u, v) in edge_set2 or (v, u) in edge_set2:
                continue
            
            # Tentatively add
            adj2[u].append(v)
            adj2[v].append(u)
            deg2[u] += 1
            deg2[v] += 1
            
            # Check validity for all edges involving u or v
            valid = True
            for x in adj2[u]:
                if x == v:
                    continue
                # Check edge (u, x)
                degs_u = set()
                for w in adj2[u]:
                    if w != x:
                        degs_u.add(deg2[w])
                degs_x = set()
                for w in adj2[x]:
                    if w != u:
                        degs_x.add(deg2[w])
                if degs_u & degs_x:
                    valid = False
                    break
                # Check edge (v, x) if exists
                if x in adj2[v]:
                    degs_v = set()
                    for w in adj2[v]:
                        if w != x:
                            degs_v.add(deg2[w])
                    degs_x2 = set()
                    for w in adj2[x]:
                        if w != v:
                            degs_x2.add(deg2[w])
                    if degs_v & degs_x2:
                        valid = False
                        break
            if valid:
                for y in adj2[v]:
                    if y == u:
                        continue
                    # Check edge (v, y)
                    degs_v = set()
                    for w in adj2[v]:
                        if w != y:
                            degs_v.add(deg2[w])
                    degs_y = set()
                    for w in adj2[y]:
                        if w != v:
                            degs_y.add(deg2[w])
                    if degs_v & degs_y:
                        valid = False
                        break
            
            if not valid:
                adj2[u].remove(v)
                adj2[v].remove(u)
                deg2[u] -= 1
                deg2[v] -= 1
            else:
                edge_set2.add((u, v))
        
        ec2 = edge_count(adj2)
        if ec2 > best_edges:
            best_edges = ec2
            best_adj = [list(a) for a in adj2]
            best_deg = list(deg2)
            best_edge_set = set(edge_set2)
    
    # Try to improve best solution by adding missing edges
    if time.time() < deadline - 2:
        for u in range(n):
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
                else:
                    best_adj[u].remove(v)
                    best_adj[v].remove(u)
                    best_deg[u] -= 1
                    best_deg[v] -= 1
    
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
