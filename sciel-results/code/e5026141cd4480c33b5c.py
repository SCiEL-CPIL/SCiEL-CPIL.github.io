import time
import random
import sys
import numpy as np

_START_TIME = time.time()
DEADLINE = 145.0  # Leave 5s margin for 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Return a dict with 'n' and 'edges' for a K4-free graph with no independent 6-set.
    Strategy: Simulated Annealing with NumPy bitsets on a fixed n (starting high, e.g., 35).
    Objective: Minimize (# K4s) + (large penalty * # Independent 6-sets).
    If we reach 0 violations, we have a valid graph.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_n = 0
    best_edges = []
    
    # We aim for n=35 (Exoo's record) or higher.
    # Starting at n=35 is hard. Let's try to construct n=35, 36, 37...
    # But SA is better at finding a valid graph for a fixed n than growing one.
    # If we can find a valid graph for n=35, score is 35/seed.
    # If we can find n=36, score is 36/seed.
    
    # Let's try a range of n values, starting from the known record and going up.
    # If we fail to find a valid graph for n, we try n-1, etc.
    # However, checking validity for large n is expensive if done naively.
    # With bitsets, checking for K4 is fast. Checking for Ind 6 is also fast with bitsets.
    
    # Strategy:
    # 1. Try n = 36, 37, 38, 39, 40 (potential new records)
    # 2. If fail, try n = 35 (known record)
    # 3. If fail, try n = 34, 33...
    # 4. Return the best valid graph found.
    
    # To save time, we will run SA for a specific n.
    # We need a fast "violation count" function.
    
    for target_n in [36, 37, 35, 34, 33, 32]:
        if _time_left() < 10:
            break
        
        # Try to find a valid graph of size target_n
        res = _solve_sa(target_n, time_budget=min(_time_left() - 5, 30))
        if res is not None:
            n, edges = res
            if n > best_n:
                best_n = n
                best_edges = edges
            # If we found a valid graph for n > 35, that's a huge win.
            # If we found 35, that matches the record.
            # We should stop if we find 36+ or if time is tight.
            if n >= 36:
                break
            # If we found 35, we might still try to find 36 with remaining time?
            # No, SA for 36 is harder. If we failed 36 earlier, we likely won't succeed now.
            # But maybe the first attempt was unlucky. Let's try one more time for 36 if we have time?
            # Actually, the loop tries 36 first. If it failed, we moved to 35.
            # If 35 succeeds, we return it.
            break
        
    # Fallback: If SA failed completely, use a deterministic construction or smaller graph
    if best_n == 0:
        # Paley graph on 17
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 17 in squares]
        best_n = n
    else:
        # Ensure we have edges for the best_n found
        pass # _solve_sa returns edges

    return {'n': best_n, 'edges': best_edges}

def _solve_sa(n, time_budget=20.0):
    """
    Run Simulated Annealing to find a K4-free graph with no independent 6-set on n vertices.
    Returns (n, edges) if found, else None.
    """
    if n < 4:
        return (n, [])
    
    # Initialize adjacency matrix as numpy uint8 or bool
    # Use uint8 for fast bitwise ops if needed, but bool is fine for logic.
    # Actually, for bitset intersection, we want integers.
    # Let's represent each vertex's neighborhood as a Python int (bitmask).
    # n <= 40 fits easily in a Python int.
    
    # Init: Random graph with p=0.5
    # Or, start with a sparse graph?
    # K4-free suggests sparse. Ind 6-free suggests dense.
    # Random regular is good. Let's start with p=0.5 and let SA fix it.
    
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and rng.random() < 0.5:
                mask |= (1 << j)
        adj[i] = mask
    
    # Precompute masks for vertices
    # We need to count K4s and Ind 6s.
    
    def count_violations(adj):
        """
        Returns (num_k4, num_ind6).
        """
        num_k4 = 0
        num_ind6 = 0
        
        # Count K4s:
        # A K4 is a set {a,b,c,d} where all pairs are edges.
        # Efficient way: For each edge (u,v), count common neighbors w such that w is connected to u and v.
        # Then for each triangle (u,v,w), check if there is a 4th vertex x connected to u,v,w.
        # Simpler: Iterate over all pairs (u,v) that are edges.
        # common = adj[u] & adj[v]
        # For each w in common, the number of K4s containing u,v,w is popcount(adj[u] & adj[v] & adj[w]).
        # But this counts each K4 4 times (once for each vertex).
        # So total K4s = sum(popcount(adj[u] & adj[v] & adj[w])) / 4 for all triangles (u,v,w).
        
        # Let's just count triangles first? No, direct K4 count.
        # For each pair (u, v) with u < v and edge exists:
        #   common = adj[u] & adj[v]
        #   For each w in common with w > v:
        #       # (u,v,w) is a triangle
        #       # Count x in common & adj[w] with x > w
        #       # This counts each K4 exactly once?
        #       # Vertices u < v < w < x.
        #       # Edges (u,v), (u,w), (u,x), (v,w), (v,x), (w,x) must exist.
        #       # adj[u] & adj[v] & adj[w] contains x.
        #       # Yes, this iterates over sorted vertices and counts each K4 once.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Iterate w > v
                    # w must be in common
                    # We can iterate over bits in common that are > v
                    # But we also need w < x.
                    # So:
                    # For each w in common with w > v:
                    #   For each x in (common & adj[w]) with x > w:
                    #       num_k4 += 1
                    
                    # Optimization:
                    # Let C = common.
                    # We want to count triples (w, x) such that v < w < x, w in C, x in C, x in adj[w].
                    # This is equivalent to counting edges in the induced subgraph on C,
                    # but restricted to vertices > v, and counting each edge once?
                    # No, (w,x) is an edge if x in adj[w].
                    # So for a fixed u,v, the number of K4s is the number of edges in the subgraph induced by C \ {0..v},
                    # where the edges are defined by adj.
                    # Wait, if x in C and x in adj[w], then (w,x) is an edge.
                    # So we just need to count the number of edges in the induced subgraph of C with vertices > v.
                    
                    # Let's extract the subgraph on vertices > v.
                    # Mask for vertices > v:
                    mask_gt_v = ((1 << n) - 1) ^ ((1 << (v + 1)) - 1)
                    C_sub = common & mask_gt_v
                    
                    # Count edges in induced subgraph of C_sub.
                    # An edge (w, x) with w < x exists if (adj[w] >> x) & 1.
                    # We can iterate w in C_sub:
                    #   for x in C_sub & adj[w] & mask_gt_w:
                    #       count += 1
                    
                    # This is O(n^2) per pair (u,v) in worst case, which is O(n^4) total.
                    # For n=35, 35^4 = 1.5M. This is fast in Python? Maybe.
                    # But we call this in SA loop. 1.5M * 1000 steps = 1.5B ops. Too slow.
                    
                    # We need a faster violation counter.
                    # Alternative: Track the number of K4s incrementally?
                    # Or, just check for *existence* of K4 or Ind 6, not count?
                    # SA needs a cost function. Existence (0/1) is a step function, bad for SA.
                    # But we can use "number of minimal K4s" or similar.
                    
                    # Let's optimize the counting.
                    # For n=35, the number of pairs is ~600.
                    # If we do O(n) work per pair, total O(n^3) = 40k ops per eval.
                    # 1000 evals = 40M ops. This is okay in Python if optimized.
                    
                    # How to count edges in induced subgraph of C_sub in O(n)?
                    # We can't easily do it in O(n) without precomputation.
                    # However, we can just check for *any* K4.
                    # If we find one, cost = 1.
                    # If we find one Ind 6, cost = 10.
                    # This is a "hard" constraint approach.
                    # SA with hard constraints:
                    # If violation > 0, accept move only if it reduces violation.
                    # If violation == 0, accept move if it keeps violation 0.
                    # This is basically local search / hill climbing.
                    # To escape local optima, we allow moves that increase violation slightly?
                    # No, if we have a K4, we must remove it.
                    
                    # Let's stick to counting, but optimize.
                    # Actually, for n=35, O(n^4) per step is too slow for 100s.
                    # We need O(n^3) or better.
                    
                    # Let's change strategy:
                    # Instead of counting all K4s, just find ONE K4 and ONE Ind 6.
                    # Cost = (1 if K4 else 0) + (10 if Ind6 else 0).
                    # This is a valid cost function for "repair".
                    # But it's flat. SA might get stuck.
                    # Better: Cost = number of "bad" edges?
                    
                    # Let's try a different metric:
                    # For each vertex, count "K4-contributions"?
                    
                    # Let's go back to counting but use NumPy?
                    # No, bitmasks are Python ints.
                    
                    # Let's try to make the counting faster.
                    # For each pair (u,v), count edges in C_sub.
                    # We can precompute for each vertex w, the list of neighbors > w?
                    # No, adj changes.
                    
                    # Okay, let's use the "find one" approach but with a twist:
                    # We will use a "potential" function that is easier to compute.
                    # Potential = sum over all vertices v of (deg(v) - 1) * (something)?
                    
                    # Actually, let's just use the "find one" approach and rely on random restarts / simulated annealing with "free" moves when cost is 0.
                    # If cost > 0, we ONLY accept moves that reduce cost.
                    # If cost == 0, we accept moves that keep cost 0.
                    # This is deterministic hill climbing. To escape, we need to allow "worse" moves.
                    # But "worse" means introducing a K4 or Ind 6.
                    # If we introduce a K4, we must remove it.
                    # This is effectively a "repair" phase.
                    
                    # Let's implement a hybrid:
                    # 1. Hill climb to remove violations.
                    # 2. If stuck (no move reduces violation), try a random "kick" (flip a random edge, even if it increases violation) and then hill climb again.
                    
                    # This is "Iterated Local Search" (ILS).
                    
                    # Let's implement ILS with fast "find violation" checks.
        
        # ILS Strategy:
        # 1. Start with random graph.
        # 2. Repair loop:
        #    - Find a K4 or Ind 6.
        #    - If found, try flipping edges to remove it.
        #    - If no violation, we have a valid graph!
        # 3. If stuck in repair (can't remove violation without creating another?),
        #    - Perturb: flip 1-5 random edges.
        #    - Go back to repair.
        
        # We need fast "find K4" and "find Ind 6".
        
        def find_k4(adj):
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        continue
                    common = adj[u] & adj[v]
                    # Check if any w > v in common has a neighbor x > w in common & adj[w]
                    # Actually, just check if common & adj[w] has any bit set for w in common.
                    # For each w in common with w > v:
                    #   if (common & adj[w]) & mask_gt_w: return (u,v,w,x)
                    # We can just check if (common & adj[w]) != 0 for any w > v in common?
                    # No, x must be > w.
                    # So (common & adj[w]) & mask_gt_w != 0.
                    
                    # Iterate w in common with w > v
                    w_mask = common & mask_gt_v
                    while w_mask:
                        # Extract lowest bit
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        # Check if there is x > w in common & adj[w]
                        if (common & adj[w]) & mask_gt_w:
                            # Find x
                            x_mask = (common & adj[w]) & mask_gt_w
                            x_lsb = x_mask & -x_mask
                            x = x_lsb.bit_length() - 1
                            return (u, v, w, x)
                        w_mask ^= lsb
            return None
        
        def find_ind6(adj):
            # Ind 6 in G = Clique 6 in Complement.
            # Complement adj: ~adj[i] & mask_all
            # But we can just check for independent set directly.
            # Independent set: no edges between them.
            # For u < v < w < x < y < z:
            #   No edges between any pair.
            
            # Fast check:
            # For each u:
            #   non_neighbors = ~adj[u] & mask_all & mask_gt_u
            #   For each v in non_neighbors:
            #     non_neighbors_v = non_neighbors & ~adj[v] & mask_gt_v
            #     ...
            
            # This is O(n^6) worst case, but with pruning it's fast.
            # For n=35, it's usually very fast unless the graph is dense (many non-neighbors).
            # If graph is dense, Ind 6 is rare, so we fail fast.
            # If graph is sparse, K4 is rare, so we fail fast on K4.
            
            for u in range(n):
                # Candidates for v: > u, not connected to u
                cands = (~adj[u]) & mask_all & mask_gt_u
                while cands:
                    lsb = cands & -cands
                    v = lsb.bit_length() - 1
                    cands2 = cands & (~adj[v]) & mask_gt_v
                    while cands2:
                        lsb2 = cands2 & -cands2
                        w = lsb2.bit_length() - 1
                        cands3 = cands2 & (~adj[w]) & mask_gt_w
                        while cands3:
                            lsb3 = cands3 & -cands3
                            x = lsb3.bit_length() - 1
                            cands4 = cands3 & (~adj[x]) & mask_gt_x
                            while cands4:
                                lsb4 = cands4 & -cands4
                                y = lsb4.bit_length() - 1
                                cands5 = cands4 & (~adj[y]) & mask_gt_y
                                while cands5:
                                    lsb5 = cands5 & -cands5
                                    z = lsb5.bit_length() - 1
                                    return (u, v, w, x, y, z)
                                cands5 ^= lsb5
                            cands4 ^= lsb4
                        cands3 ^= lsb3
                    cands2 ^= lsb2
                cands ^= lsb
            return None
        
        mask_all = (1 << n) - 1
        # Precompute mask_gt_i for speed?
        mask_gt = [((1 << n) - 1) ^ ((1 << (i + 1)) - 1) for i in range(n)]
        
        def is_valid(adj):
            return find_k4(adj) is None and find_ind6(adj) is None
        
        # ILS Loop
        best_adj = None
        start_time = time.time()
        max_time = time_budget
        
        # Initial random graph
        adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and rng.random() < 0.5:
                    mask |= (1 << j)
            adj[i] = mask
        
        # Check if valid
        if is_valid(adj):
            return _adj_to_edges(adj, n)
        
        # ILS
        while time.time() - start_time < max_time:
            # Repair Phase
            # Try to fix violations
            progress = True
            repair_iters = 0
            while progress and repair_iters < 1000:
                progress = False
                repair_iters += 1
                
                k4 = find_k4(adj)
                if k4:
                    u, v, w, x = k4
                    # Try to break this K4 by removing one edge
                    # Edges: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
                    pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    # Try each removal, see if it reduces violations (or at least doesn't increase Ind6 count significantly?)
                    # Simple: Try all 6, pick the one that results in valid graph or fewer K4s?
                    # Just try random one? No, try all and pick best.
                    best_new_adj = None
                    best_score = float('inf')
                    for p in pairs:
                        a, b = p
                        # Remove edge
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        # Check validity
                        # We don't want to run full is_valid for each of 6 options if it's slow.
                        # But is_valid is fast.
                        if is_valid(adj):
                            best_new_adj = adj[:]
                            best_score = 0
                            break
                        else:
                            # Count K4s? Just check if this K4 is gone?
                            # If we remove (a,b), this specific K4 is gone.
                            # But maybe another K4 remains.
                            # Let's just count K4s and Ind6s roughly?
                            # No, let's just keep the first valid one.
                            # If none valid, we might want to pick the one that minimizes "damage".
                            # For now, just revert and try another?
                            pass
                        
                        # Revert
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                    
                    if best_new_adj is not None:
                        adj = best_new_adj
                        progress = True
                    else:
                        # No single edge removal fixed the graph.
                        # Try a random removal to break the loop?
                        p = pairs[rng.randrange(6)]
                        a, b = p
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        progress = True
                    continue
                
                ind6 = find_ind6(adj)
                if ind6:
                    u, v, w, x, y, z = ind6
                    # Add an edge to break the independent set
                    pairs = [(u,v), (u,w), (u,x), (u,y), (u,z),
                             (v,w), (v,x), (v,y), (v,z),
                             (w,x), (w,y), (w,z),
                             (x,y), (x,z),
                             (y,z)]
                    best_new_adj = None
                    for p in pairs:
                        a, b = p
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        
                        if is_valid(adj):
                            best_new_adj = adj[:]
                            break
                        else:
                            pass
                        # Revert
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    
                    if best_new_adj is not None:
                        adj = best_new_adj
                        progress = True
                    else:
                        p = pairs[rng.randrange(15)]
                        a, b = p
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        progress = True
                    continue
                
                # No K4, no Ind 6
                # Valid!
                return _adj_to_edges(adj, n)
            
            # Perturbation Phase (if repair got stuck)
            # Flip 1-5 random edges
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                # Flip edge (u,v)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if valid after perturbation
            if is_valid(adj):
                return _adj_to_edges(adj, n)
        
        return None

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return (n, edges)

# Entry point
def construct():
    """
    Return a dict with 'n' and 'edges' for a K4-free graph with no independent 6-set.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_n = 0
    best_edges = []
    
    # Try n=36 first (new record), then 35 (record), then lower.
    # We allocate time based on difficulty.
    # n=36 is harder, so give it more time?
    # Or just try them in order with fixed budgets.
    
    targets = [36, 37, 35, 34, 33, 32, 31, 30]
    
    # Dynamic time allocation
    # If we find 36, we are done.
    # If we don't, we try 35.
    
    for target_n in targets:
        if _time_left() < 10:
            break
        
        # Budget for this n
        # n=36: 30s
        # n=35: 20s
        # n<35: 10s
        if target_n >= 36:
            budget = min(30, _time_left() - 5)
        elif target_n == 35:
            budget = min(20, _time_left() - 5)
        else:
            budget = min(10, _time_left() - 5)
            
        if budget < 5:
            continue
            
        res = _solve_sa(target_n, time_budget=budget)
        if res is not None:
            n, edges = res
            if n > best_n:
                best_n = n
                best_edges = edges
            if n >= 36:
                break
            # If we found 35, we can stop unless we want to try 36 again?
            # We already tried 36 first.
            break

    if best_n == 0:
        # Fallback: Paley graph on 17
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 17 in squares]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
