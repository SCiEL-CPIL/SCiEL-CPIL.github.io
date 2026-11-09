import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 16
    # Adjacency matrix for fast lookups
    # adj[u][v] is 1 if edge exists
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u][v] = 1
            adj[v][u] = 1

    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u][v] = 0
            adj[v][u] = 0

    def has_q3():
        """
        Checks if the current graph contains a subgraph isomorphic to Q3.
        Q3 is the 3-cube: 8 vertices, each degree 3, bipartite, diameter 3.
        A subgraph isomorphic to Q3 must have 8 vertices with specific connectivity.
        Since n=16 is small, we can check all 8-vertex subsets? 
        C(16,8) = 12870. For each subset, check if induced subgraph is Q3?
        Or check if there exists a Q3 subgraph (not necessarily induced).
        
        Actually, a faster check:
        Q3 has 12 edges. 
        We can iterate over all potential "centers" or structures.
        
        Let's use a more direct structural check.
        A Q3 subgraph consists of 8 vertices.
        Let's try to find 4 disjoint edges that form the "vertical" axis? No.
        
        Efficient check for n=16:
        Iterate over all subsets of 8 vertices? 12870 subsets.
        For each subset S of 8 vertices:
          1. Check if |E(S)| == 12.
          2. Check if all vertices in S have degree 3 within S.
          3. Check if S is connected.
          4. Check if S is bipartite with partition size 4,4.
          5. Check if diameter is 3 (or just check it's not a 4-cycle with chords etc).
          Actually, if it's 3-regular, connected, bipartite 8-vertex graph, it is Q3.
          (The only 3-regular bipartite graph on 8 vertices is Q3).
          
        So:
        1. Generate all 8-subsets of vertices.
        2. For each subset, count internal edges. If != 12, skip.
        3. Check degrees within subset are all 3.
        4. Check connectivity (BFS).
        5. If passes, it's a Q3.
        
        Complexity: 12870 * (8*8) ~ 1M ops. Very fast.
        """
        # Pre-compute list of all 8-subsets for speed? 
        # Or just generate on the fly. 
        # We can optimize by only checking if we add an edge that *might* create a Q3.
        # But for simplicity and robustness in 110s, full check is fine if we don't do it too often.
        
        # Optimization: Only check if we added an edge recently?
        # No, let's just run the full check. It's fast enough.
        
        # To speed up, we can pre-compute the list of 8-subsets once outside?
        # But this function is called inside a loop. 
        # Let's generate subsets on the fly but break early.
        
        # A faster check:
        # Any Q3 contains 4 disjoint edges? No.
        # Let's stick to the subset check but optimize the inner loop.
        
        # We can prune: if any vertex in the subset has degree < 3 in the whole graph, 
        # it can't be in a Q3 subgraph where it has degree 3? 
        # No, degree in subgraph <= degree in graph. So if deg_graph(v) < 3, v cannot be in Q3.
        
        # Candidates for Q3 vertices: vertices with degree >= 3.
        # If fewer than 8 such vertices, no Q3.
        
        candidates = [i for i in range(n) if sum(adj[i]) >= 3]
        if len(candidates) < 8:
            return False
            
        # Iterate over subsets of candidates of size 8
        # If candidates > 8, we still have many subsets.
        # But if the graph is dense, candidates is likely all 16.
        
        for subset in combinations(candidates, 8):
            s = set(subset)
            # Check degrees within subset
            valid = True
            edge_count = 0
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and adj[u][v]:
                        deg += 1
                        # Avoid double counting edges
                        if u < v:
                            edge_count += 1
                if deg != 3:
                    valid = False
                    break
            if not valid:
                continue
            if edge_count != 12:
                continue
            
            # Check connectivity
            # BFS from subset[0]
            visited = {subset[0]}
            queue = [subset[0]]
            while queue:
                u = queue.pop(0)
                for v in subset:
                    if v not in visited and adj[u][v]:
                        visited.add(v)
                        queue.append(v)
            if len(visited) != 8:
                continue
                
            # If it's 3-regular, connected, 8 vertices, 12 edges -> it is Q3
            return True
            
        return False

    def try_add_edge(u, v):
        add_edge(u, v)
        if has_q3():
            remove_edge(u, v)
            return False
        return True

    # Initialization: Start with a random graph or empty?
    # Start with empty is safe but slow to build.
    # Start with a known Q3-free graph? 
    # The seed K3,13 + clique is 42 edges.
    # Let's start with a random sparse graph and grow it.
    # Or start with the seed graph and try to add more.
    
    # Let's start with the seed graph from the current program to ensure we beat it.
    # Seed: A={0,1,2}, B={3..15}. Edges: all A-B, all A-A.
    # This is 42 edges.
    
    # Re-initialize adj and edges
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    # Add seed edges
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    current_edges = len(edges)
    best_edges = current_edges
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Local Search: Try to add random edges
    # We can also try to remove edges to allow adding more? 
    # Simulated Annealing or Simple Hill Climbing.
    # Let's do Hill Climbing with random restarts if stuck.
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    # Iterate through edges, try to add them
    # If we can't add an edge, we might need to remove one.
    # For now, simple greedy addition.
    
    while time.time() < deadline:
        # Try to add a random edge
        # Pick a random edge not in graph
        # If graph is dense, this is slow. 
        # Better: maintain list of missing edges.
        
        # For simplicity, just pick random u, v
        progress = False
        for _ in range(100): # Try 100 random additions per loop iteration
            if time.time() > deadline:
                break
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            if (u, v) in edges: continue
            
            if try_add_edge(u, v):
                progress = True
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [row[:] for row in adj]
                    best_edge_set = set(edges)
        
        # If no progress in 100 tries, we are stuck.
        # Try a more systematic approach or restart?
        # Let's try to add edges in a systematic order if stuck.
        if not progress:
             # Try systematic pass
             for u in range(n):
                 for v in range(u+1, n):
                     if (u, v) not in edges:
                         if try_add_edge(u, v):
                             current_edges += 1
                             if current_edges > best_edges:
                                 best_edges = current_edges
                                 best_adj = [row[:] for row in adj]
                                 best_edge_set = set(edges)
                             progress = True
             if not progress:
                 # Stuck completely. 
                 # We could try to remove an edge to allow more?
                 # This is complex. Let's just break and return best.
                 # Or restart with a different random seed?
                 # Given time constraints, returning best found is safer.
                 # However, we can try a few restarts with random initial graphs?
                 # No, we started from a good seed. 
                 # Let's try to remove a random edge and then try to add different ones?
                 # This is "kick" in simulated annealing.
                 
                 # Try removing a random edge
                 if edges:
                    rem_u, rem_v = random.choice(list(edges))
                    remove_edge(rem_u, rem_v)
                    current_edges -= 1
                    # Now try to add random edges again
                    # The loop will continue and try to add
                    # This allows escaping local minima
                    # We don't update best here, only when we add
                    continue
        
        # If we made progress, continue loop
    
    # Return the best graph found
    result_edges = [list(e) for e in best_edge_set]
    return result_edges

