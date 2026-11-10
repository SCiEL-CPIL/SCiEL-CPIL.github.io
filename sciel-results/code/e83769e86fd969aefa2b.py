# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses an exact Branch and Bound (Tomita) algorithm to find the Maximum Independent Set.
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build graph: nodes 0..204. Edge between u, v if (u-v) mod 205 is in forbidden.
    # We want an Independent Set in this graph, which is a set with no forbidden differences.
    
    # adjacency list for fast neighbor iteration
    # adj[u] is a list of v such that (u-v) mod 205 is forbidden
    adj = [[] for _ in range(m)]
    for u in range(m):
        for v in range(u + 1, m):
            d1 = (u - v) % m
            d2 = (v - u) % m
            # Since forbidden is symmetric (if d is square, -d is square mod p? 
            # Actually x^2 = y^2 mod m doesn't imply -x^2 is square. 
            # Wait, F = {x^2 mod 205}. 
            # If d in F, is m-d in F? 
            # 205 = 5 * 41. 
            # Squares mod 5: {0, 1, 4}. -1 = 4 is square.
            # Squares mod 41: 1, 4, 16, 25, 36, 9, 28, 2, 38, 10, 33, 17, 35, 20, 8, 30, 15, 32, 21, 7. 
            # -1 mod 41 = 40. Is 40 a square? 40 = -1. Legendre symbol (-1/41) = 1 since 41=1 mod 4. 
            # So yes, -1 is a square mod 41. 
            # By CRT, -1 is a square mod 205. 
            # So if d is in F, m-d is in F.
            # Thus the graph is undirected.
            if d1 in forbidden:
                adj[u].append(v)
                adj[v].append(u)
    
    # Sort adjacency lists to help with ordering in Tomita? 
    # Tomita usually benefits from ordering vertices by degree or just using a specific heuristic.
    # Let's use a simple greedy initial bound first.
    
    def greedy_mis():
        """Greedy construction of an independent set."""
        # Order vertices by degree (ascending) is often good for greedy MIS
        # Or just random/sequential. Let's try degree ascending.
        degrees = [len(adj[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: degrees[x])
        
        selected = []
        selected_set = set()
        
        for u in order:
            if u in selected_set:
                continue
            # Check if u conflicts with any selected node
            conflict = False
            for v in selected:
                if (u - v) % m in forbidden:
                    conflict = True
                    break
            if not conflict:
                selected.append(u)
                selected_set.add(u)
        
        return selected

    initial_solution = greedy_mis()
    best_size = len(initial_solution)
    best_set = initial_solution[:]

    # Tomita's Branch and Bound for Maximum Independent Set
    # We work on the subgraph induced by the candidate vertices.
    # To make it efficient, we maintain a list of candidates.
    
    # We'll use a recursive function that takes:
    # 1. list of candidate vertices (P)
    # 2. current independent set (I)
    # 3. size of current set
    # 4. best size found so far
    
    # Optimization: We can order P. A common heuristic is to pick the vertex with the smallest degree in the induced subgraph of P.
    
    # To avoid recursion depth issues and overhead, we'll implement it carefully.
    # N=205 is small enough for recursion if we prune well.
    
    # Precompute a bitmask or just use sets? Sets are fine for N=205.
    # But list operations are faster.
    
    # Let's use a global timeout guard
    start_time = time.time()
    deadline = start_time + 590
    timeout_flag = [False]
    
    def check_timeout():
        if time.time() > deadline:
            timeout_flag[0] = True
            return True
        return False

    def tomita(p_list, current_set, current_size):
        nonlocal best_size, best_set, timeout_flag
        
        if timeout_flag[0]:
            return
            
        if not p_list:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Pruning: if current_size + len(p_list) <= best_size, no point continuing
        if current_size + len(p_list) <= best_size:
            return
            
        # Heuristic: Pick a vertex u from p_list.
        # Strategy: Choose u with minimum degree in the subgraph induced by p_list.
        # This is expensive to compute from scratch. 
        # Alternative: Just pick the first one, or one with low global degree.
        # Let's stick to a simpler heuristic: pick the vertex with the fewest neighbors in p_list.
        
        # To speed up, we can just iterate and pick one.
        # For N=205, even O(N^2) per node is okay if tree is pruned.
        
        # Find u in p_list with min degree in p_list
        min_deg = float('inf')
        u_idx = 0
        p_set = set(p_list) # For fast neighbor check? 
        # Actually, checking neighbors in p_list:
        # Neighbors of u in p_list are v in p_list such that v in adj[u]
        
        # Let's just pick the first vertex for simplicity first, then refine if too slow.
        # Actually, Tomita's algorithm specifically uses the "Coloring" bound (MCS) to prune.
        # Implementing MCS is complex. 
        # Given N=205, a simple Branch and Bound with greedy upper bound (len(p_list)) might be enough if the graph is sparse/dense appropriately.
        # The graph of squares mod 205: 
        # Number of squares is approx 205/2 = 102.
        # Degree of each node is approx 102.
        # This is a fairly dense graph.
        # Independent sets will be small.
        
        # Let's try a recursive search picking the vertex with the smallest degree in the induced subgraph.
        
        # Calculate degrees in induced subgraph
        # p_list is a list of ints.
        # To find degree of u in p_list: count how many v in p_list are in adj[u].
        # This is O(|p_list| * deg(u)) if we use a set for p_list.
        
        p_set = set(p_list)
        min_deg = 1000
        u = p_list[0]
        for v in p_list:
            deg = 0
            for w in adj[v]:
                if w in p_set:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                u = v
        
        # Branch 1: Include u
        # New candidates: neighbors of u? No, independent set means we exclude neighbors of u.
        # New P = P \ ({u} U N(u))
        new_p = []
        u_neighbors = set(adj[u])
        for v in p_list:
            if v != u and v not in u_neighbors:
                new_p.append(v)
        
        new_set = current_set + [u]
        tomita(new_p, new_set, current_size + 1)
        
        if timeout_flag[0]:
            return
            
        # Branch 2: Exclude u
        new_p2 = [v for v in p_list if v != u]
        tomita(new_p2, current_set, current_size)

    # Run Tomita
    # Initial P is all vertices
    p_init = list(range(m))
    
    # To improve pruning, we can sort initial P? 
    # The heuristic inside Tomita handles ordering dynamically.
    
    # One optimization: If we find a solution of size k, we can prune branches where potential size < k.
    # The `current_size + len(p_list) <= best_size` check does this.
    
    # Let's run it.
    # To avoid stack overflow, we might need to increase recursion limit, but 205 depth is fine.
    import sys
    sys.setrecursionlimit(10000)
    
    tomita(p_init, [], 0)
    
    # If timeout occurred, best_set might be from initial greedy or partial search.
    # Ensure we return a valid set.
    # The tomita algorithm only updates best_set when it finds a *better* valid set.
    # So best_set is always valid.
    
    # If for some reason best_set is empty (shouldn't happen), fallback to [0]
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
