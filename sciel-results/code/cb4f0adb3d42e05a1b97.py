import random
import time

def construct():
            # 1. Define Grötzsch graph
            cycle = [(i, (i + 1) % 5) for i in range(5)]
            base_edges = [list(e) for e in cycle]
            for u, w in cycle:
                base_edges += [[u, 5 + w], [w, 5 + u]]
            base_edges += [[5 + u, 10] for u in range(5)]
            
            # 2. Setup for extension
            n = 16
            adj = [[0]*n for _ in range(n)]
            edge_set = set()
            
            for u, v in base_edges:
                adj[u][v] = 1
                adj[v][u] = 1
                edge_set.add((min(u,v), max(u,v)))
            
            # 3. Generate all possible edges not in base
            possible_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edge_set:
                        possible_edges.append((i,j))
            
            # 4. Greedy addition with restarts
            best_edges = len(base_edges)
            best_adj = [row[:] for row in adj]
            
            start_time = time.time()
            deadline = 110 # Safety margin
            
            while time.time() < deadline:
                # Shuffle possible edges
                random.shuffle(possible_edges)
                
                curr_adj = [row[:] for row in best_adj]
                curr_edge_count = best_edges
                
                for u, v in possible_edges:
                    if curr_adj[u][v]:
                        continue
                    
                    # Check if adding (u,v) creates a triangle
                    # Triangle exists if there is w such that adj[u][w] and adj[v][w]
                    is_triangle = False
                    for w in range(n):
                        if w != u and w != v:
                            if curr_adj[u][w] and curr_adj[v][w]:
                                is_triangle = True
                                break
                    
                    if not is_triangle:
                        curr_adj[u][v] = 1
                        curr_adj[v][u] = 1
                        curr_edge_count += 1
                
                if curr_edge_count > best_edges:
                    best_edges = curr_edge_count
                    best_adj = [row[:] for row in curr_adj]
            
            # 5. Reconstruct edge list
            final_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if best_adj[i][j]:
                        final_edges.append([i, j])
                        
            return final_edges
