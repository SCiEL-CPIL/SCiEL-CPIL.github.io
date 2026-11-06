import numpy as np
import time
import random

def construct():
    # Target n to maximize. We'll try to reach as high as possible.
    # Start with a reasonable guess, e.g., 30, and try to increase if time permits?
    # Or just start high and fix?
    # Starting high (e.g., 35) with a good initial density is better.
    
    n = 35
    # Time limit is 120s. We use 110s for search.
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial graph: Random with p=0.5
    # Adjacency matrix as boolean numpy array
    A = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(A, False)
    # Generate random edges
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                A[i, j] = True
                A[j, i] = True

    def find_clique_size_k(adj, k):
        """
        Finds a clique of size k in the graph represented by adj (bool matrix).
        Returns list of vertices if found, else None.
        Uses simple backtracking with bitset pruning if possible, 
        but for k=4,6 and n=35, simple recursion might be okay if optimized.
        """
        vertices = list(range(n))
        
        def backtrack(current_clique, candidates):
            if len(current_clique) == k:
                return current_clique[:]
            if len(candidates) < k - len(current_clique):
                return None
            
            # Heuristic: pick candidate with highest degree in induced subgraph?
            # For simplicity, just iterate.
            # Optimization: If candidates is small, check directly.
            
            for i in range(len(candidates)):
                v = candidates[i]
                # Check if v is connected to all in current_clique
                if all(adj[v, u] for u in current_clique):
                    # New candidates are those in candidates[i+1:] connected to v
                    new_candidates = [u for u in candidates[i+1:] if adj[v, u]]
                    res = backtrack(current_clique + [v], new_candidates)
                    if res:
                        return res
            return None

        # Initial candidates: all vertices
        # To speed up, we can order vertices by degree
        degrees = np.sum(adj, axis=1)
        ordered_vertices = [v for v in range(n) if degrees[v] >= k-1]
        if not ordered_vertices:
            return None
            
        # Sort by degree descending to find cliques faster?
        ordered_vertices.sort(key=lambda v: degrees[v], reverse=True)
        
        return backtrack([], ordered_vertices)

    def find_independent_set_size_k(adj, k):
        """
        Finds an independent set of size k in adj.
        Equivalent to finding a clique of size k in the complement.
        """
        # Complement adjacency (excluding self)
        # We don't want to build full complement matrix every time if possible,
        # but for n=35, it's cheap.
        comp = ~adj
        np.fill_diagonal(comp, False)
        return find_clique_size_k(comp, k)

    def get_violation(adj):
        """
        Returns (k4_found, ind6_found).
        """
        k4 = find_clique_size_k(adj, 4)
        ind6 = find_independent_set_size_k(adj, 6)
        return k4, ind6

    # Main Search Loop
    best_valid_n = 0
    best_graph = None
    
    # We will try to fix the graph of size n.
    # If we succeed, we can try n+1? 
    # The task says "Maximize n". 
    # It's safer to find a valid graph for a specific n.
    # Let's try to find a valid graph for n=35.
    # If we find it, we return it.
    # If not, we might return a valid graph for a smaller n if we found one during restarts?
    # The seed returns a specific n. 
    # The score is n. 
    # So we should aim for the highest n we can make valid.
    
    # Strategy: 
    # 1. Try n=35. 
    # 2. If fail, try n=34, etc? 
    # Or just restart n=35 with different seeds.
    # Given time constraints, we'll stick to one n per "session" but maybe restart multiple times.
    
    # Let's do multiple restarts for n=35.
    # If we find a valid graph, return it immediately.
    
    max_restarts = 1000
    restart = 0
    
    while time.time() < deadline and restart < max_restarts:
        restart += 1
        # Re-initialize or perturb?
        # Full re-initialization is safer to escape bad local minima.
        # We can vary the initial density p.
        p_init = 0.4 + 0.2 * (restart % 3) / 2.0 # 0.4, 0.5, 0.6
        
        A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p_init:
                    A[i, j] = True
                    A[j, i] = True

        # Local search
        steps = 0
        max_steps = 5000
        last_violation_count = -1
        stuck_counter = 0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            k4, ind6 = get_violation(A)
            
            if k4 is None and ind6 is None:
                # Valid graph found!
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Decide which violation to fix
            # If both exist, pick one randomly or the one that is "easier" (smaller subgraph)
            # K4 has 6 edges, Ind6 has 0 edges (in G, 15 in comp).
            # Removing an edge from K4 is always a valid move to break it.
            # Adding an edge to Ind6 is always a valid move to break it.
            
            if k4 is not None and ind6 is not None:
                if random.random() < 0.5:
                    target = k4
                    is_k4 = True
                else:
                    target = ind6
                    is_k4 = False
            elif k4 is not None:
                target = k4
                is_k4 = True
            else:
                target = ind6
                is_k4 = False

            # Pick a random edge in the violating subgraph to flip
            if is_k4:
                # K4 vertices: v0, v1, v2, v3
                # Edges: (v0,v1), (v0,v2), (v0,v3), (v1,v2), (v1,v3), (v2,v3)
                v = target
                edges_in_clique = [
                    (v[0], v[1]), (v[0], v[2]), (v[0], v[3]),
                    (v[1], v[2]), (v[1], v[3]), (v[2], v[3])
                ]
                u, w = random.choice(edges_in_clique)
                # Remove edge
                A[u, w] = False
                A[w, u] = False
            else:
                # Ind 6 vertices: v0..v5
                # We need to ADD an edge between any pair in this set.
                v = target
                # Generate all pairs
                pairs = []
                for i in range(6):
                    for j in range(i+1, 6):
                        pairs.append((v[i], v[j]))
                u, w = random.choice(pairs)
                # Add edge
                A[u, w] = True
                A[w, u] = True
            
            # Simple acceptance: always accept (Hill Climbing on violation count)
            # To avoid cycles, we could track history, but randomization helps.
            # If we are stuck (same violation count for many steps), we can do a random jump.
            # For now, just continue.
            
            # Optional: If we detect a cycle (repeating states), break and restart.
            # Hard to track state easily.
            
            # If time is running out, break
            if time.time() > deadline - 5:
                break

    # If we didn't find a valid graph for n=35, 
    # we should return the best valid graph we found for a smaller n?
    # The problem asks to maximize n. 
    # If we can't validate 35, maybe we should try 34?
    # However, the loop above restarts. 
    # If we exit the loop without returning, we must return *something* valid.
    # The seed returned a valid graph. 
    # We must ensure we return a valid graph.
    
    # Fallback: Return the seed graph (Paley 17) if we fail?
    # Or try to find a valid graph for n=17, 18, etc.
    # Let's add a fallback loop for smaller n if 35 fails.
    
    # Actually, the probability of finding a valid graph for n=35 with simple local search is low.
    # We should probably lower the target n to ensure we get a valid graph, 
    # or improve the search.
    
    # Let's try n=30 first, which is easier.
    # If we find n=30, score is 30/17 = 1.76.
    # If we find n=35, score is 35/17 = 2.05.
    
    # Let's modify the loop to try a sequence of n: 35, 34, 33...
    # But we only have 120s.
    
    # Revised Plan:
    # 1. Try n=35 with aggressive local search.
    # 2. If fail, try n=30.
    # 3. If fail, return seed.
    
    # But the code above only tries n=35.
    # Let's wrap it in a function that takes n.

    def try_find_graph(n_target, time_limit):
        start = time.time()
        # ... (move logic here) ...
        # For now, I will implement the logic for a single n in the main construct,
        # but I'll change n to 30 to be safer, or keep 35 and accept lower success rate?
        # The prompt says "Maximize n". 
        # If I return n=17, score is 1.0.
        # If I return n=22, score is 1.29.
        # If I return n=30, score is 1.76.
        
        # Let's aim for 30. It's much more likely to be found.
        
        pass

    # Let's rewrite the main logic to try n=30.
    
    n_target = 30
    # Re-use the logic above with n = n_target
    
    # I will implement the search for n_target directly.
    
    # Reset time tracking for this specific attempt?
    # No, we have global deadline.
    
    # Let's do 5 restarts for n=30.
    
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        p_init = 0.5
        A = np.zeros((n_target, n_target), dtype=bool)
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < p_init:
                    A[i, j] = True
                    A[j, i] = True
        
        steps = 0
        max_steps = 2000
        while steps < max_steps and time.time() < deadline - 10:
            steps += 1
            k4, ind6 = get_violation(A)
            if k4 is None and ind6 is None:
                edges = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if A[i, j]:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
            
            if k4 is not None and ind6 is not None:
                if random.random() < 0.5:
                    target, is_k4 = k4, True
                else:
                    target, is_k4 = ind6, False
            elif k4 is not None:
                target, is_k4 = k4, True
            else:
                target, is_k4 = ind6, False

            if is_k4:
                v = target
                edges_in_clique = [
                    (v[0], v[1]), (v[0], v[2]), (v[0], v[3]),
                    (v[1], v[2]), (v[1], v[3]), (v[2], v[3])
                ]
                u, w = random.choice(edges_in_clique)
                A[u, w] = False
                A[w, u] = False
            else:
                v = target
                pairs = [(v[i], v[j]) for i in range(6) for j in range(i+1, 6)]
                u, w = random.choice(pairs)
                A[u, w] = True
                A[w, u] = True

    # If we failed to find a valid graph for n=30, return the seed graph (n=17)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
