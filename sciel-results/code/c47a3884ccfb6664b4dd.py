import numpy as np
import heapq
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Convert to numpy bool array for fast operations
    adj = np.asarray(adjacency, dtype=bool)
    
    # Original degrees
    orig_deg = adj.sum(axis=1).astype(int)
    
    # We want to compute a score that favors vertices that are "peripheral" 
    # in the graph structure. The greedy decoder picks vertices in descending 
    # score order. So we want high scores for vertices that are safe to pick early.
    
    # Structural features:
    # 1. Core number: Lower core number means the vertex is in a "softer" part of the graph.
    # 2. Peeling order: The order in which vertices are removed in min-degree peeling.
    # 3. Neighborhood degree: Sum of degrees of neighbors (clustering proxy).
    
    # Let's compute the core number properly using the Batagelj-Zaversnik algorithm 
    # or a simplified min-degree peeling that assigns core numbers.
    
    # For N=200, we can do a simple iterative peeling.
    # We'll simulate the k-core decomposition.
    
    # Algorithm for Core Number:
    # 1. deg[v] = degree of v
    # 2. k = max(deg)
    # 3. While k > 0:
    #    a. Find a vertex v with deg[v] == k (and not yet assigned core number)
    #       If no such vertex, k-- and continue.
    #    b. Assign core[v] = k
    #    c. For each neighbor u of v:
    #       If core[u] is unassigned:
    #          deg[u] -= 1
    #          If deg[u] == k:
    #             # u is now a candidate for core number k
    #             # We don't immediately assign it, we just note it.
    #             # But the standard algorithm requires careful handling.
    
    # Let's use the heap-based peeling which gives a "peeling order" and "peeling degree".
    # The "peeling degree" (degree at time of removal) is often a good proxy for core number.
    # Actually, the core number of v is the MAXIMUM peeling degree of any vertex that was removed 
    # BEFORE v in the peeling process? No, that's not right.
    
    # Let's just use the peeling order and the degree at removal.
    # Vertices removed early have low "coreness".
    
    degs = orig_deg.copy()
    heap = [(degs[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = [False] * n
    peel_order = np.zeros(n, dtype=int)
    peel_deg = np.zeros(n, dtype=int)
    step = 0
    
    while heap:
        if time.time() > deadline:
            break
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if degs[v] != d:
            continue
        removed[v] = True
        peel_order[v] = step
        peel_deg[v] = d
        step += 1
        
        neighbors = np.where(adj[v, :])[0]
        for u in neighbors:
            if not removed[u]:
                degs[u] -= 1
                heapq.heappush(heap, (degs[u], u))
    
    # If the heap broke early due to deadline, fill in missing values with worst-case scores
    # to ensure we return a valid array.
    if step < n:
        for i in range(n):
            if not removed[i]:
                peel_order[i] = n - 1
                peel_deg[i] = 0
    
    # Now, how to combine these into a score?
    # We want high score for vertices that are likely to be in a large independent set.
    # Vertices with low degree are generally easier to include.
    # Vertices in low cores are in "loose" parts of the graph.
    
    # A good heuristic is:
    # Score = -peel_order (lower order is better)
    # But peel_order alone might not capture the "density" of the neighborhood.
    
    # Let's try a composite score:
    # 1. Normalize peel_order to [0, 1] (0 is first removed, 1 is last)
    # 2. Normalize peel_deg to [0, 1] (0 is low degree at removal, 1 is high)
    # 3. Score = - (alpha * norm_peel_order + beta * norm_peel_deg)
    
    # However, the greedy decoder skips neighbors. 
    # If we pick a vertex with low degree, we block few others.
    # If we pick a vertex with high degree, we block many others.
    # So, low original degree is a strong signal.
    
    # Let's test a simple combination:
    # Score = -orig_deg - 0.1 * peel_order
    
    # Or, we can use the "core number" explicitly.
    # The core number can be approximated by the maximum peel_deg seen so far?
    # No, let's compute the actual core number using a simple method for N=200.
    
    # Actually, let's look at the "branching factor" idea from the inspiration.
    # "Sum of weights derived from its branching factor in a k-core decomposition"
    # "Exponentially decaying weights to vertices based on their removal order"
    
    # Let's try:
    # Score[v] = sum_{u in neighbors of v} exp(-lambda * peel_order[u])
    # This rewards vertices whose neighbors are removed LATE (high coreness).
    # Wait, if neighbors are removed late, they are "central". 
    # If I pick a vertex whose neighbors are central, I block central vertices?
    # That seems bad.
    
    # Conversely, if neighbors are removed EARLY (low coreness), they are peripheral.
    # So I should prefer vertices whose neighbors are peripheral?
    # That would mean picking central vertices?
    
    # Let's rethink.
    # If I pick vertex v, I gain 1, and I lose all neighbors of v.
    # Net gain = 1 - sum_{u in N(v)} (probability u would have been picked if v wasn't there)
    # This is hard to estimate.
    
    # Heuristic: Pick vertices with low degree first.
    # Among low degree vertices, pick those in low cores.
    
    # Let's use:
    # Score = -orig_deg * 2.0 - peel_deg * 1.0 - peel_order * 0.01
    
    # Let's try to be more sophisticated.
    # Compute the "independence number" estimate?
    
    # Let's stick to a robust structural score.
    # The current program uses -peel_order.
    # The seed uses -orig_deg.
    # The current program (score 1.03) is better than seed (1.00).
    # So -peel_order is better than -orig_deg?
    
    # Let's try combining them.
    # Normalize both to [0, 1].
    
    max_po = np.max(peel_order)
    if max_po > 0:
        norm_po = peel_order / max_po
    else:
        norm_po = np.zeros(n)
        
    max_pd = np.max(peel_deg)
    if max_pd > 0:
        norm_pd = peel_deg / max_pd
    else:
        norm_pd = np.zeros(n)
        
    max_od = np.max(orig_deg)
    if max_od > 0:
        norm_od = orig_deg / max_od
    else:
        norm_od = np.zeros(n)
        
    # Score = -(w1 * norm_od + w2 * norm_pd + w3 * norm_po)
    # Let's try w1=0.5, w2=0.3, w3=0.2
    
    scores = -(0.5 * norm_od + 0.3 * norm_pd + 0.2 * norm_po)
    
    return scores
