import time
import random
import math

N = 15
DEADLINE = 85.0  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    
    def find_violations(adj, deg):
        """
        Returns a list of tuples (edge1, edge2, edge3) for every simple path s-u-v-t 
        where deg(s) == deg(t).
        Also returns a dictionary counting how many violations each edge is part of.
        """
        violations = []
        edge_counts = { (i, j): 0 for i in range(N) for j in range(i+1, N) }
        
        # Iterate over all ordered triples of distinct vertices u, v, w to form paths?
        # A path is s-u-v-t.
        # We can iterate u, v (adjacent), then s in N(u)\{v}, t in N(v)\{u}.
        # Ensure s != t.
        
        for u in range(N):
            for v in range(N):
                if u == v or not adj[u][v]:
                    continue
                # s is neighbor of u, not v
                # t is neighbor of v, not u
                # s != t
                
                # Optimization: Pre-compute neighbors lists
                # But N is small, direct iteration is fine.
                
                for s in range(N):
                    if s == u or s == v or not adj[u][s]:
                        continue
                    for t in range(N):
                        if t == u or t == v or t == s or not adj[v][t]:
                            continue
                        # Path s-u-v-t is simple and exists.
                        if deg[s] == deg[t]:
                            violations.append((s, u, v, t))
                            # Increment counts for edges (s,u), (u,v), (v,t)
                            e1 = (min(s, u), max(s, u))
                            e2 = (min(u, v), max(u, v))
                            e3 = (min(v, t), max(v, t))
                            edge_counts[e1] += 1
                            edge_counts[e2] += 1
                            edge_counts[e3] += 1
                            
        return violations, edge_counts

    def greedy_remove(adj, deg, neighbors, seed):
        """
        Start from current state, remove edges to fix violations.
        Returns the resulting edge count.
        """
        random.seed(seed)
        
        # Copy state
        adj_c = [row[:] for row in adj]
        deg_c = deg[:]
        neigh_c = [set(n) for n in neighbors]
        
        # We need a mutable edge set or just use adj_c
        # Let's track edges as a set of tuples for easy removal and iteration
        current_edges = set()
        for u in range(N):
            for v in range(u+1, N):
                if adj_c[u][v]:
                    current_edges.add((u, v))
        
        max_iterations = 200 # Safety cap
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Find violations
            violations, edge_counts = find_violations(adj_c, deg_c)
            
            if not violations:
                break
            
            # Find edge with max count
            max_count = 0
            candidates = []
            for e in current_edges:
                c = edge_counts.get(e, 0)
                if c > max_count:
                    max_count = c
                    candidates = [e]
                elif c == max_count and c > 0:
                    candidates.append(e)
            
            if not candidates:
                # Should not happen if violations exist, but just in case
                break
                
            # Pick one randomly
            e_to_remove = random.choice(candidates)
            u, v = e_to_remove
            
            # Remove edge
            adj_c[u][v] = 0
            adj_c[v][u] = 0
            deg_c[u] -= 1
            deg_c[v] -= 1
            neigh_c[u].remove(v)
            neigh_c[v].remove(u)
            current_edges.remove(e_to_remove)
            
        return len(current_edges)

    # Initialize with K_N
    init_adj = [[1] * N for _ in range(N)]
    for i in range(N):
        init_adj[i][i] = 0
        
    init_deg = [N - 1] * N
    init_neighbors = [set(range(N)) for _ in range(N)]
    for i in range(N):
        init_neighbors[i].remove(i)
        
    best_count = 0
    best_adj = None
    best_deg = None
    
    num_restarts = 200
    seed = 42
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        count = greedy_remove(init_adj, init_deg, init_neighbors, seed + i)
        
        if count > best_count:
            best_count = count
            # We need to reconstruct the best_adj to return it.
            # The greedy_remove function modifies local copies. 
            # To save time, we can just run the greedy process again to get the final state 
            # or store the state in the function. 
            # Given N=15, re-running is fast. But let's modify greedy_remove to return the final state if it's the best.
            # Actually, simpler: just re-run the best seed at the end to get the exact adj matrix.
            # Or better, modify greedy_remove to return the final adj.
            
    # Re-run the best strategy to get the exact edges
    # We need to know which seed was best. Let's store the best seed.
    
    # Let's refactor slightly: run greedy_remove, if it beats best, save the seed.
    
    best_seed = -1
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        count = greedy_remove(init_adj, init_deg, init_neighbors, seed + i)
        
        if count > best_count:
            best_count = count
            best_seed = seed + i

    # Reconstruct best graph
    if best_seed != -1:
        # We need a version of greedy_remove that returns the final adj matrix
        # Let's write a helper that returns the final state
        def greedy_remove_state(adj, deg, neighbors, s):
            random.seed(s)
            adj_c = [row[:] for row in adj]
            deg_c = deg[:]
            neigh_c = [set(n) for n in neighbors]
            current_edges = set()
            for u in range(N):
                for v in range(u+1, N):
                    if adj_c[u][v]:
                        current_edges.add((u, v))
            
            for _ in range(200):
                if time.time() - start_time > DEADLINE - 2:
                    break
                violations, edge_counts = find_violations(adj_c, deg_c)
                if not violations:
                    break
                max_count = 0
                candidates = []
                for e in current_edges:
                    c = edge_counts.get(e, 0)
                    if c > max_count:
                        max_count = c
                        candidates = [e]
                    elif c == max_count and c > 0:
                        candidates.append(e)
                if not candidates:
                    break
                e_to_remove = random.choice(candidates)
                u, v = e_to_remove
                adj_c[u][v] = 0
                adj_c[v][u] = 0
                deg_c[u] -= 1
                deg_c[v] -= 1
                neigh_c[u].remove(v)
                neigh_c[v].remove(u)
                current_edges.remove(e_to_remove)
                
            return adj_c, deg_c, current_edges

        final_adj, final_deg, final_edges = greedy_remove_state(init_adj, init_deg, init_neighbors, best_seed)
        
        # Final verification
        v, _ = find_violations(final_adj, final_deg)
        if v:
            # Fallback: empty graph
            return []
            
        result = []
        for u in range(N):
            for v_ in range(u + 1, N):
                if final_adj[u][v_]:
                    result.append([int(u), int(v_)])
        return result
    else:
        return []

if __name__ == "__main__":
    # Test
    edges = construct()
    print(len(edges))
