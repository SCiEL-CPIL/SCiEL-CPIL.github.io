import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    # Partitions for bipartite structure
    L = list(range(10))
    R = list(range(10, 20))
    
    # We will try to construct a dense bipartite graph first.
    # Then add intra-partition edges.
    
    def get_bipartite_edges():
        edges = []
        for u in L:
            for v in R:
                edges.append((u, v))
        return edges
    
    def get_intra_edges():
        edges = []
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                edges.append((L[i], L[j]))
        for i in range(len(R)):
            for j in range(i+1, len(R)):
                edges.append((R[i], R[j]))
        return edges
    
    # Local search for bipartite coloring
    # State: dict edge -> color
    # Cost: number of 4-cycles with < 4 distinct colors
    
    def count_conflicts_bipartite(edge_colors):
        # edge_colors: dict (u, v) -> c
        # 4-cycle: u1, v1, u2, v2 with u1,u2 in L, v1,v2 in R
        # Edges: (u1, v1), (u2, v1), (u1, v2), (u2, v2)
        # Check if all 4 exist and colors are distinct
        conflicts = 0
        # Precompute adjacency for fast lookup
        # adj[u] = {v: c}
        adj = {}
        for (u, v), c in edge_colors.items():
            adj.setdefault(u, {})[v] = c
            adj.setdefault(v, {})[u] = c
            
        for u1 in L:
            for u2 in L:
                if u1 >= u2: continue
                for v1 in R:
                    for v2 in R:
                        if v1 >= v2: continue
                        
                        # Check if all 4 edges exist
                        if v1 not in adj.get(u1, {}) or v2 not in adj.get(u1, {}) or \
                           v1 not in adj.get(u2, {}) or v2 not in adj.get(u2, {}):
                            continue
                            
                        c1 = adj[u1][v1]
                        c2 = adj[u2][v1]
                        c3 = adj[u1][v2]
                        c4 = adj[u2][v2]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            conflicts += 1
        return conflicts

    best_bip_edges = None
    best_bip_conflicts = float('inf')
    
    # Initialize with random colors
    bip_edges = get_bipartite_edges()
    current_colors = {}
    for e in bip_edges:
        current_colors[e] = random.choice(colors)
        
    # Simulated annealing / Random restarts for bipartite
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
            
        # Random initialization
        for e in bip_edges:
            current_colors[e] = random.choice(colors)
            
        conflicts = count_conflicts_bipartite(current_colors)
        if conflicts == 0:
            best_bip_edges = dict(current_colors)
            break
            
        # Local search
        temp = 1.0
        for step in range(10000):
            if time.time() - start_time > DEADLINE:
                break
                
            e = random.choice(bip_edges)
            old_c = current_colors[e]
            new_c = random.choice(colors)
            if old_c == new_c:
                continue
                
            current_colors[e] = new_c
            new_conflicts = count_conflicts_bipartite(current_colors)
            
            if new_conflicts < conflicts or random.random() < pow(2.718, -(new_conflicts - conflicts) / temp):
                conflicts = new_conflicts
                if conflicts == 0:
                    break
            else:
                current_colors[e] = old_c
                
            temp *= 0.999
            
        if conflicts < best_bip_conflicts:
            best_bip_conflicts = conflicts
            best_bip_edges = dict(current_colors)
            
        if best_bip_conflicts == 0:
            break

    # Build the graph from best_bip_edges
    # If we didn't find a perfect bipartite, we might still have a good partial one?
    # No, the cost function counts conflicts in the full K_10,10.
    # If we can't get 0 conflicts, we can't use K_10,10.
    # We should fall back to a greedy approach if bipartite fails.
    
    final_edges = []
    adj = {}
    
    def add_edge(u, v, c):
        final_edges.append([u, v, c])
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c
        
    def check_validity_incremental(u, v, c, adj):
        """Check if adding edge (u,v) with colour c violates any 4-cycle constraint."""
        vs = list(adj.get(v, {}).keys())
        for x in vs:
            if x == u:
                continue
            xs = list(adj.get(x, {}).keys())
            for y in xs:
                if y == u or y == v or y == x:
                    continue
                if u not in adj.get(y, {}):
                    continue
                c_vx = adj[v][x]
                c_xy = adj[x][y]
                c_yu = adj[y][u]
                cols = [c, c_vx, c_xy, c_yu]
                if len(set(cols)) != 4:
                    return False
        return True

    if best_bip_edges and best_bip_conflicts == 0:
        # We have a valid K_10,10
        for e, c in best_bip_edges.items():
            u, v = e
            add_edge(u, v, c)
            
        # Now try to add intra-partition edges
        intra_edges = get_intra_edges()
        random.shuffle(intra_edges)
        
        for u, v in intra_edges:
            if time.time() - start_time > DEADLINE:
                break
            trial_colors = list(range(n))
            random.shuffle(trial_colors)
            for c in trial_colors:
                if check_validity_incremental(u, v, c, adj):
                    add_edge(u, v, c)
                    break
    else:
        # Fallback: Greedy construction on all edges
        # Use the logic from Inspiration 1 which was successful
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        # Prioritize bipartite edges in the shuffle?
        # Just shuffle all
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() - start_time > DEADLINE:
                break
            trial_colors = list(range(n))
            random.shuffle(trial_colors)
            for c in trial_colors:
                if check_validity_incremental(u, v, c, adj):
                    add_edge(u, v, c)
                    break
    
    # Final validation
    def _validate(edges):
        adj_check = {}
        for u, v, c in edges:
            adj_check.setdefault(u, {})[v] = c
            adj_check.setdefault(v, {})[u] = c
        from itertools import combinations
        verts = list(range(n))
        for a, b, c, d in combinations(verts, 4):
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)],
            ]
            for cyc in cycles:
                cols = []
                valid_cycle = True
                for (u, v) in cyc:
                    if v not in adj_check.get(u, {}):
                        valid_cycle = False
                        break
                    cols.append(adj_check[u][v])
                if valid_cycle:
                    if len(set(cols)) != 4:
                        return False
        return True

    if not _validate(final_edges):
        # Fallback to a safe sparse graph
        return [[0, v, 0] for v in range(1, 20)]
        
    return final_edges

if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
