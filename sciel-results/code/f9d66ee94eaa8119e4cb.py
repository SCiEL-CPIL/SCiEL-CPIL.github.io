import time
import random
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # 10s margin
        n = 20
        best_edges = 0
        best_graph = None
        
        # Helper to check validity of a specific edge addition or whole graph
        def get_common_neighbors(adj, i, j, k):
            return adj[i] & adj[j] & adj[k]
            
        def is_valid_graph(adj):
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        if bin(common).count('1') >= 3:
                            return False
            return True

        def find_k33(adj):
            # Returns a list of edges forming a K3,3 or None
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        # Need 3 bits set
                        if bin(common).count('1') >= 3:
                            # Extract 3 neighbors
                            neighbors = []
                            c = common
                            while len(neighbors) < 3 and c:
                                lsb = c & -c
                                neighbors.append(lsb.bit_length() - 1)
                                c -= lsb
                            # Edges are (i,x), (j,x), (k,x) for x in neighbors
                            # Actually K3,3 is between {i,j,k} and {x,y,z}
                            # So edges are all pairs from {i,j,k} x {x,y,z}
                            edges = []
                            for u in [i,j,k]:
                                for v in neighbors:
                                    edges.append((u,v))
                            return edges
            return None

        def purge(adj):
            # Make graph valid
            while True:
                k33_edges = find_k33(adj)
                if not k33_edges:
                    break
                # Remove a random edge from the K3,3
                u, v = random.choice(k33_edges)
                if u > v: u, v = v, u
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        def greedy_add(adj):
            # Try to add all missing edges
            improved = True
            while improved:
                improved = False
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] >> v) & 1:
                            # Check if adding (u,v) keeps validity
                            # Only check triples containing u or v
                            valid = True
                            # Triples with u
                            for x in range(n):
                                if x == u: continue
                                for y in range(x+1, n):
                                    if y == u: continue
                                    if x == v or y == v:
                                         # Triple is (u, x, y) or (u, v, x) etc
                                         # Actually, just check all triples involving u or v
                                         pass
                            # Simpler: Just check the whole graph?
                            # n=20, 1140 triples.
                            # 1140 * 20 * 20 = 456,000 checks per edge addition?
                            # No, 1140 checks per graph state.
                            # We are adding one edge at a time.
                            # We can just tentatively add, check, revert.
                            
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
                            if is_valid_graph(adj):
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                
        def local_search(adj):
            # 1-2 swaps to escape local optima
            # Try to add edge. If fail, swap.
            while time.time() < deadline:
                # Try to add a random missing edge
                added = False
                missing_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] >> v) & 1:
                            missing_edges.append((u,v))
                            
                if missing_edges:
                    random.shuffle(missing_edges)
                    for u, v in missing_edges:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_valid_graph(adj):
                            added = True
                            break
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            
                if added:
                    continue
                    
                # Stuck. Try 1-2 swap
                # Pick random existing edge to remove
                existing_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            existing_edges.append((u,v))
                            
                if not existing_edges:
                    break
                    
                # Try a few swaps
                for _ in range(100):
                    if not existing_edges: break
                    u1, v1 = random.choice(existing_edges)
                    if not missing_edges: break
                    u2, v2 = random.choice(missing_edges)
                    
                    # Remove e1, Add e2
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid_graph(adj):
                        # Update edge lists
                        existing_edges.remove((u1,v1))
                        missing_edges.remove((u2,v2))
                        # Also need to update other lists? 
                        # This is getting complex. 
                        # Maybe simpler: Just restart the greedy add phase?
                        break
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)

        # Main Loop
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            # Initialize random graph
            p = 0.4 + 0.2 * (restarts % 3) / 2.0 # Vary density
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            purge(adj)
            greedy_add(adj)
            
            # Count edges
            edge_count = sum(bin(x).count('1') for x in adj) // 2
            
            if edge_count > best_edges:
                best_edges = edge_count
                # Save state
                best_adj = adj[:]
                
            # Local search to improve current graph
            # To avoid infinite loops, limit local search time per restart
            ls_start = time.time()
            # Simple local search: try adding edges, if stuck, swap
            # We will just run greedy_add again after some random swaps?
            # Let's do a proper local search loop
            
            while time.time() < deadline and (time.time() - ls_start) < 10:
                # Try to add
                added = False
                missing = [(u,v) for u in range(n) for v in range(u+1,n) if not (adj[u]>>v)&1]
                if missing:
                    random.shuffle(missing)
                    for u,v in missing:
                        adj[u] |= (1<<v)
                        adj[v] |= (1<<u)
                        if is_valid_graph(adj):
                            added = True
                            break
                        else:
                            adj[u] &= ~(1<<v)
                            adj[v] &= ~(1<<u)
                if added:
                    continue
                    
                # Swap
                existing = [(u,v) for u in range(n) for v in range(u+1,n) if (adj[u]>>v)&1]
                if not existing or not missing:
                    break
                for _ in range(50):
                    u1,v1 = random.choice(existing)
                    u2,v2 = random.choice(missing)
                    adj[u1] &= ~(1<<v1)
                    adj[v1] &= ~(1<<u1)
                    adj[u2] |= (1<<v2)
                    adj[v2] |= (1<<u2)
                    if is_valid_graph(adj):
                        # Update lists? No, just break and re-calculate
                        break
                    else:
                        adj[u1] |= (1<<v1)
                        adj[v1] |= (1<<u1)
                        adj[u2] &= ~(1<<v2)
                        adj[v2] &= ~(1<<u2)
                        
            edge_count = sum(bin(x).count('1') for x in adj) // 2
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = adj[:]

        # Convert best_adj to list of edges
        edges = []
        if best_adj:
            for u in range(n):
                for v in range(u+1, n):
                    if (best_adj[u] >> v) & 1:
                        edges.append([u, v])
        else:
            # Fallback
            edges = [[a, b] for a in range(2) for b in range(2, 20)]
            
        return edges
