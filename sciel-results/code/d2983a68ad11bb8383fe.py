import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    def is_c4_free(adj):
        """
        Check if the graph represented by adjacency matrix 'adj' is C4-free.
        A graph is C4-free if and only if no two vertices have more than 1 common neighbor.
        This is equivalent to (A @ A)[i, j] <= 1 for all i != j.
        """
        if adj.shape[0] == 0:
            return True
        A = adj.astype(int)
        A2 = A @ A
        # Diagonal elements of A2 are degrees, which can be > 1.
        # We only care about off-diagonal elements.
        mask = ~np.eye(adj.shape[0], dtype=bool)
        if np.any(A2[mask] >= 2):
            return False
        return True

    def get_min_degree(adj):
        if adj.size == 0:
            return 0
        return int(np.min(adj.sum(axis=1)))

    def greedy_boost(adj, n, target_deg, time_limit):
        """
        Try to add edges to increase the minimum degree to target_deg,
        maintaining the C4-free property.
        """
        adj = adj.copy()
        degrees = adj.sum(axis=1).astype(int)
        
        # We will repeatedly find a vertex with degree < target_deg
        # and try to connect it to a non-neighbor with minimal common neighbors.
        
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Find vertices with degree < target
            deficient = [i for i in range(n) if degrees[i] < target_deg]
            if not deficient:
                break
            
            # Pick a deficient vertex (randomly to avoid bias, or min degree)
            # Random helps escape local minima in connectivity structure
            u = random.choice(deficient)
            
            # Find non-neighbors of u
            non_neighbors = [v for v in range(n) if v != u and not adj[u, v]]
            if not non_neighbors:
                # u is connected to everyone. If degree is still low, it's impossible 
                # (degree would be n-1). But if n is small, maybe?
                # For n=46, max degree is 45. Target is 7. So this shouldn't happen 
                # unless we are very dense.
                continue
            
            # Find v in non_neighbors that minimizes common neighbors with u
            # If min common neighbors <= 1, we can add edge (u, v).
            min_common = n
            best_v = -1
            
            # To speed up, we can just check a random subset if list is large, 
            # but n=46 is small.
            random.shuffle(non_neighbors)
            
            for v in non_neighbors:
                # Count common neighbors
                # adj[u] and adj[v] are boolean arrays
                common = np.sum(adj[u] & adj[v])
                if common < min_common:
                    min_common = common
                    best_v = v
                    if min_common == 0:
                        break
            
            if best_v != -1 and min_common <= 1:
                # Add edge
                adj[u, best_v] = True
                adj[best_v, u] = True
                degrees[u] += 1
                degrees[best_v] += 1
            else:
                # Stuck on this u? Try another deficient vertex in next iteration.
                # To prevent infinite loop on the same stuck u, 
                # we rely on random.choice picking different vertices.
                pass
        
        return adj

    def generate_field_graph(n, time_limit):
        """
        Generate a C4-free graph on n vertices using finite field structure.
        We use the fact that in GF(p^k), the equation x^2 + y^2 = 1 has 
        specific properties. 
        However, a simpler robust method is:
        1. Start with a known C4-free graph (e.g. polarity graph of PG(2,q) if n matches, 
           or a random C4-free graph).
        2. Here we use a "Random C4-Free" generator:
           - Start with empty graph.
           - Add edges randomly if they don't create a C4.
           - This tends to produce a graph with good expansion.
        """
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # List of all possible edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        # We want to add as many edges as possible while keeping C4-free.
        # But we specifically want to ensure high min degree.
        # So we prioritize adding edges to low-degree vertices.
        
        # Strategy: 
        # Iterate through edges. If adding an edge keeps C4-free, add it.
        # But to boost min degree, we should check edges incident to low-degree vertices first.
        
        # Let's do a multi-pass approach.
        # Pass 1: Add any safe edge.
        # Pass 2: Add safe edges incident to min-degree vertices.
        
        # Actually, just adding edges in random order often results in a regular-ish graph
        # because the "constraint" of no C4 limits local clustering, spreading edges out.
        
        added = 0
        for u, v in edges:
            if time.time() > time_limit:
                break
            
            # Check if adding (u, v) creates a C4.
            # Condition: common_neighbors(u, v) <= 1.
            common = np.sum(adj[u] & adj[v])
            if common <= 1:
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
                added += 1
        
        return adj

    def solve_for_n(n, time_limit):
        """
        Try to construct a valid graph for a given n.
        Returns (adj, min_deg) if successful, else (None, 0).
        """
        target_deg = n - 39
        
        # Multiple restarts with different random seeds/strategies
        best_adj = None
        best_deg = 0
        
        for attempt in range(10):
            if time.time() > time_limit:
                break
            
            random.seed(attempt * 12345 + n)
            
            # Generate a base C4-free graph
            # Using random greedy edge addition
            base_adj = generate_field_graph(n, time_limit)
            
            # Check current min degree
            curr_deg = get_min_degree(base_adj)
            
            if curr_deg >= target_deg:
                return base_adj, curr_deg
            
            # Try to boost min degree
            boosted_adj = greedy_boost(base_adj, n, target_deg, time_limit)
            
            new_deg = get_min_degree(boosted_adj)
            
            if new_deg > best_deg:
                best_deg = new_deg
                best_adj = boosted_adj
            
            if best_deg >= target_deg:
                return best_adj, best_deg
        
        if best_deg >= target_deg:
            return best_adj, best_deg
        return None, best_deg

    # Try n=47
    n_try = 47
    adj, deg = solve_for_n(n_try, deadline - 10)
    if adj is not None:
        edges = []
        for i in range(n_try):
            for j in range(i+1, n_try):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n_try, 'edges': edges}

    # Try n=46
    n_try = 46
    adj, deg = solve_for_n(n_try, deadline - 10)
    if adj is not None:
        edges = []
        for i in range(n_try):
            for j in range(i+1, n_try):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n_try, 'edges': edges}
    
    # Fallback: Try n=45, 44, etc., down to 39
    for n_try in range(45, 38, -1):
        if time.time() > deadline - 5:
            break
        target_deg = n_try - 39
        # A trivial graph: disjoint union of cliques? No, C4-free.
        # Just use random greedy.
        adj, deg = solve_for_n(n_try, deadline - 5)
        if adj is not None:
            edges = []
            for i in range(n_try):
                for j in range(i+1, n_try):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n_try, 'edges': edges}
            
    # Last resort: n=39, empty graph
    return {'n': 39, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
