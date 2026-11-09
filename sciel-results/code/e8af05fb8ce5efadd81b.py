import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # 1. Generate Candidate Edges
    # The graph must be triangle-free. The base path is bipartite (Even/Odd).
    # Adding an edge between two vertices of the same parity creates a triangle 
    # if they are connected via the path? No, an edge (u,v) with u,v same parity 
    # creates a cycle of length (v-u)+1. If v-u is odd, cycle length is even.
    # Wait. Path is 0-1-2. Edge (0,2) creates triangle 0-1-2-0.
    # So any edge (u,v) where (v-u) is even creates a triangle with the path segment.
    # Therefore, ALL added edges MUST be between Even and Odd vertices.
    # This preserves the bipartite structure of the graph.
    # A bipartite graph is always triangle-free.
    
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    candidates = []
    for e in even_verts:
        for o in odd_verts:
            if abs(e - o) == 1:
                continue # Already in base
            candidates.append((e, o))
            
    # Sort candidates by "potential" - edges that bridge large gaps first?
    # Or just random? Let's sort by distance descending to prioritize "long" shortcuts.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    best_solution = None
    best_size = n # Upper bound: path needs n-1 edges, we can always do worse
    
    # We want to minimize added edges.
    # Let's try to find a solution with k edges, starting from a lower bound.
    # Lower bound for diameter 4 on 24 nodes?
    # Star graph needs 23 edges.
    # Hub at 12 needed 10 edges.
    # Let's try k from 1 to 10.
    
    def bfs_diameter(adj):
        # Returns diameter or -1 if disconnected
        diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return -1 # Early exit if diameter > 4
                        q.append(v)
            if -1 in dist:
                return -1 # Disconnected
            max_d = max(dist)
            if max_d > diam:
                diam = max_d
        return diam
    
    def build_adj(added_edges):
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Backtracking search
    # State: current set of added edges, index in candidates
    # We will iterate through candidates and decide to include or exclude.
    # To make it efficient, we can use a greedy lower bound or just prune if 
    # current_size + remaining_potential < best_size? No, we are minimizing size.
    # We want to find ANY solution of size k.
    # Let's use a recursive backtracking that tries to add edges.
    
    # Optimization: Pre-compute adjacency for base.
    base_adj = [[] for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Since the graph is bipartite, we can represent it efficiently.
    # But for n=24, simple list of lists is fine.
    
    def solve(k):
        """
        Find a set of k edges from candidates that makes diameter <= 4.
        Returns the set of edges if found, else None.
        """
        # We will pick k edges from candidates.
        # Combinations of 200+ candidates taken k at a time is too many.
        # We need to prune.
        
        # Heuristic ordering:
        # We can use a recursive function that picks edges one by one.
        # Pruning: If the current graph (base + picked edges) has some property 
        # that prevents diameter <= 4 with only (k - len(picked)) more edges, prune.
        
        # A simpler pruning: 
        # If we pick an edge, it reduces distances.
        # Let's just do a randomized local search / simulated annealing instead?
        # No, the directive asks for a STRUCTURAL change. CSP/Backtracking is structural.
        
        # Let's try a specific structure search:
        # 1. Single Hub: We already know 10 edges works. Can we do 9?
        # 2. Two Hubs: 
        # 3. General Search
        
        # Let's implement a depth-first search with heuristics.
        # To avoid exponential blowup, we limit the depth and use randomization for ordering.
        
        # Actually, let's look at the "island 5" finding: "Greedy construction with local search (edge removal) finds minimal valid edge set".
        # My current code is just a fallback. I should implement the "Jump-based initial guess + Removal" strategy which is proven to work in this population, but I need to make it *different* from the current "Hub-and-Spoke" construction.
        # The current program constructs a specific hub solution.
        # The new strategy: 
        # 1. Generate a dense random bipartite graph (or multiple random graphs).
        # 2. Check if diameter <= 4.
        # 3. If yes, run "Edge Removal" (Local Search) to minimize edges.
        # 4. Keep the best result.
        # This is a "Random Restart + Local Search" strategy, distinct from "Constructive Heuristic".
        
        pass # Placeholder, will implement below

    # Implementation of Random Restart + Local Search (Edge Removal)
    
    import random
    random.seed(12345)
    
    def try_local_search(initial_edges, max_time=1.0):
        """
        Takes a set of edges that yields diameter <= 4.
        Attempts to remove edges to reduce count while maintaining diameter <= 4.
        """
        current_edges = list(initial_edges)
        current_set = set(current_edges)
        
        # Verify initial
        if not _is_valid(current_edges):
            return None
            
        improved = True
        while improved:
            improved = False
            # Try removing each edge
            for i in range(len(current_edges)):
                if time.time() > max_time:
                    break
                # Remove edge i
                test_edges = current_edges[:i] + current_edges[i+1:]
                if _is_valid(test_edges):
                    current_edges = test_edges
                    improved = True
                    break # Restart loop with new length
        
        return current_edges

    def _is_valid(added_edges):
        # Fast check
        adj = build_adj(added_edges)
        # Check Diameter <= 4
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
            if -1 in dist:
                return False
        return True

    best_found = None
    best_size = 11 # We know 10 works, so look for < 10
    
    # Strategy: Generate random bipartite graphs with varying density
    # and run local search.
    
    # To be efficient, we should bias the random generation towards "good" structures.
    # Good structures for diameter 4 usually involve high-degree nodes.
    
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        # Generate a random set of edges
        # Size of random set: start with something large enough to likely be connected and small diameter
        # e.g., 15-20 edges.
        num_edges = random.randint(12, 20)
        
        # To improve chances, let's ensure some "hub" connectivity.
        # Pick 2-3 random hubs and connect them to random others?
        # Or just random?
        # Let's mix: 50% pure random, 50% hub-biased.
        
        if attempt % 2 == 0:
            # Pure Random
            random_edges = random.sample(candidates, num_edges)
        else:
            # Hub Biased
            # Pick 1-2 hubs
            num_hubs = random.randint(1, 2)
            hubs = random.sample(range(n), num_hubs)
            random_edges = []
            edge_set = set()
            for h in hubs:
                # Connect hub to random vertices of opposite parity
                opp = odd_verts if h % 2 == 0 else even_verts
                targets = random.sample(opp, min(len(opp), num_edges // num_hubs + 2))
                for t in targets:
                    e = (min(h, t), max(h, t))
                    if e not in base_set and e not in edge_set:
                        random_edges.append(e)
                        edge_set.add(e)
            # Fill rest with random
            while len(random_edges) < num_edges:
                e = random.choice(candidates)
                if e not in edge_set:
                    random_edges.append(e)
                    edge_set.add(e)
        
        # Check if this random graph has diameter <= 4
        if _is_valid(random_edges):
            # Run local search to minimize
            result = try_local_search(random_edges, max_time=0.5)
            if result is not None:
                if best_found is None or len(result) < len(best_found):
                    best_found = result
                    best_size = len(result)
                    if best_size <= 8: # If we get very good, we can stop or keep looking for 7
                        # Let's keep looking for a bit to see if we can beat 9
                        pass

    # Fallback: The known 10-edge solution
    fallback = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]
    
    if best_found is not None:
        return [[int(u), int(v)] for u, v in best_found]
    else:
        return [[int(u), int(v)] for u, v in fallback]

# EVOLVE-BLOCK-END
