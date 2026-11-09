"""Rewritten program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=float)

    # Convert to numpy bool array if needed
    adj = np.asarray(adjacency, dtype=bool)
    deg = adj.sum(axis=1).astype(np.int64)

    # Deadline guard: we have ~150s total, but we want to be safe.
    # This function is called once per evaluation.
    # The core peeling and scoring is O(n^2) or O(n^3) worst case, which is tiny for n=200.
    # So we don't really need a complex deadline guard for the computation itself,
    # but we will include a simple check just in case.
    start_time = time.time()
    DEADLINE = 140.0  # 10s margin

    # 1. Core Decomposition (k-core peeling)
    # We compute the core number for each vertex.
    # Core number of v is the largest k such that v belongs to the k-core.
    # We simulate peeling: repeatedly remove vertices with minimum degree.
    # The order of removal gives us a "peeling order".
    # Vertices removed early are in low cores. Vertices removed late are in high cores.
    
    # For n=200, we can do this efficiently.
    # We maintain a list of active vertices and their current degrees.
    # We use a min-heap for (degree, vertex).
    
    import heapq
    
    active = [True] * n
    curr_deg = deg.copy()
    heap = [(curr_deg[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removal_order = [0] * n  # removal_order[i] = step at which vertex i was removed (0-indexed)
    core_number = [0] * n
    current_max_degree = 0
    step = 0
    
    # Standard k-core peeling algorithm
    # We need to be careful: the core number is the minimum degree at the time of removal?
    # Actually, the core number is the maximum k such that the vertex is in the k-core.
    # In the peeling process, when a vertex is removed, its current degree is its core number?
    # No, that's not quite right. The core number is the highest k for which it remains in the k-core.
    # In the peeling algorithm, vertices are removed in non-decreasing order of their current degree.
    # The core number of a vertex is the maximum degree it had during the peeling process?
    # Actually, a simpler way: 
    # The core number of v is the largest k such that there exists a subgraph where every vertex has degree >= k and v is in it.
    # In the peeling process, we remove vertices with the smallest current degree.
    # Let k = current degree of the vertex being removed.
    # Then the core number of that vertex is at most k.
    # Actually, the core number is exactly the degree at the time of removal?
    # Let's check. If we remove a vertex with degree d, it cannot be in the (d+1)-core.
    # It might be in the d-core.
    # So core_number[v] = degree at time of removal?
    # Wait, if I have a triangle, all degrees are 2.
    # Remove v1 (deg 2). core_number[v1] = 2?
    # Remaining edge v2-v3. Degrees are 1.
    # Remove v2 (deg 1). core_number[v2] = 1?
    # Remove v3 (deg 0). core_number[v3] = 0?
    # But the triangle is a 2-core. All vertices should have core number 2.
    # So the simple "degree at removal" is not the core number.
    
    # Correct algorithm:
    # The core number of v is the maximum k such that v is in the k-core.
    # We can compute this by noting that the core number is the maximum over all subgraphs containing v of the minimum degree in that subgraph.
    # Alternatively, we can use the property that the core number is the largest k such that v is not removed before the k-th round of peeling?
    # No.
    
    # Let's use a different approach.
    # We can compute the core numbers by iterating k from 0 to n and finding the k-core.
    # But that's O(n^2) or more.
    # For n=200, it's fine.
    
    # Actually, there is a linear time algorithm for core numbers.
    # But let's stick to a simple method for now.
    
    # Let's re-implement the peeling correctly to get core numbers.
    # The core number of a vertex v is the highest k such that v is in the k-core.
    # In the peeling process, we remove vertices in order of increasing degree.
    # Let's keep track of the degree at the time of removal, but that's not enough.
    
    # Alternative:
    # We can compute the core numbers by the following:
    # 1. Compute degrees.
    # 2. Put all vertices in a bucket based on degree.
    # 3. Iterate k from 0 to max_degree.
    # 4. While there are vertices with degree < k, remove them and update neighbors.
    # 5. The core number of a vertex is the k at which it was "finalized"?
    
    # Let's use the standard algorithm from Batagelj and Zaversnik (2003).
    # But for n=200, we can just do:
    # For each k from 0 to n:
    #   Find the k-core.
    #   If v is in the k-core, core_number[v] = max(core_number[v], k)
    
    # This is O(n * (n+m)) which is fine for n=200.
    
    core_numbers = np.zeros(n, dtype=np.int64)
    
    # We can optimize by just doing the peeling once and recording the order.
    # Actually, the core number of v is the maximum k such that v is in the k-core.
    # It is known that the core number of v is equal to the maximum degree of v in any subgraph? No.
    
    # Let's just compute it by checking for each k.
    # Since n is small, this is acceptable.
    
    # But wait, there is a simpler way.
    # The core number of v is the largest k such that there are at least k neighbors of v in the k-core?
    
    # Let's use the peeling order.
    # In the peeling process, vertices are removed in non-decreasing order of their "core number".
    # Actually, the peeling order is such that if u is removed before v, then core_number[u] <= core_number[v].
    # And core_number[v] is the maximum degree of v in the graph induced by vertices removed after v? No.
    
    # Let's just compute core numbers properly.
    # We'll use the bucket approach.
    
    max_deg = int(deg.max()) if n > 0 else 0
    
    # deg_buckets: list of lists, where deg_buckets[d] contains vertices with current degree d
    # But degrees change.
    
    # Let's just use the simple O(n^2) method for core numbers.
    # For k from 0 to max_deg:
    #   active_k = all vertices
    #   while True:
    #     find vertex v in active_k with degree < k (in the subgraph induced by active_k)
    #     if none, break
    #     remove v from active_k
    #   core_numbers[v] for v in active_k is at least k
    
    # This is a bit slow but n=200 is tiny.
    
    # Actually, let's just use the removal order from the heap.
    # The core number of a vertex is the maximum k such that it is not removed in the first k rounds?
    # No.
    
    # Let's try a different structural feature:
    # Instead of core numbers, let's use the "independent set potential".
    # A vertex with low degree is good.
    # A vertex with many low-degree neighbors is good (because picking it might not block many other good vertices? No, picking it blocks its neighbors).
    # Actually, we want to pick vertices that are "isolated" in the sense that they don't block many other good vertices.
    # So, a vertex v is good if its neighbors have high degrees (because they are less likely to be picked anyway? No, if they have high degrees, they are less likely to be picked, so picking v doesn't "waste" the chance to pick them? No, if they are not picked, then v can be picked. But if v is picked, they are blocked. But if they wouldn't be picked anyway, then blocking them is no loss).
    
    # So, a vertex v is a good candidate if:
    # 1. Its degree is low.
    # 2. Its neighbors have high degrees (so they are unlikely to be picked, meaning v doesn't conflict with other good candidates).
    
    # Let's define a score:
    # score[v] = -deg[v] + alpha * mean(deg[neighbors])
    
    # Or maybe:
    # score[v] = -deg[v] + beta * sum(1/deg[u] for u in neighbors) ?
    
    # Let's try:
    # score[v] = -deg[v] + gamma * (sum(deg[u] for u in neighbors) / deg[v]) if deg[v] > 0 else 0
    
    # This rewards vertices that have neighbors with high degrees.
    
    # Let's also consider the second order:
    # If a neighbor u has many neighbors, and those neighbors have high degrees, then u is "strong" and unlikely to be picked.
    
    # Let's try a simple combination:
    # score[v] = -deg[v] + 0.5 * mean(deg[neighbors])
    
    # But we need to be careful with deg[v] = 0.
    
    # Let's also try:
    # score[v] = -deg[v] + 0.1 * (sum(deg[u] for u in neighbors))
    
    # Let's compute a few features and combine them.
    
    # Feature 1: degree
    # Feature 2: mean neighbor degree
    # Feature 3: max neighbor degree
    # Feature 4: number of neighbors with degree < deg[v] (i.e., v is a "hub" relative to its neighbors)
    
    # Let's compute these.
    
    mean_neighbor_deg = np.zeros(n, dtype=float)
    max_neighbor_deg = np.zeros(n, dtype=float)
    
    for i in range(n):
        if deg[i] > 0:
            neighbors = np.where(adj[i])[0]
            mean_neighbor_deg[i] = deg[neighbors].mean()
            max_neighbor_deg[i] = deg[neighbors].max()
        else:
            mean_neighbor_deg[i] = 0
            max_neighbor_deg[i] = 0
    
    # Let's try a weighted sum:
    # score = -deg + 0.5 * mean_neighbor_deg
    
    # But we need to normalize? No, the greedy algorithm just sorts by score.
    
    # Let's try:
    # score = -deg + 0.5 * mean_neighbor_deg
    
    # But what if deg is very large?
    # Let's try to make it more robust.
    
    # Let's also consider the "independent set size" heuristic.
    # The expected size of the independent set if we pick v is 1 + (size of independent set in G - N[v]).
    # But we can't compute that.
    
    # Let's try a different approach:
    # Use the "local ratio" or "local density".
    
    # Let's try:
    # score[v] = 1 / (1 + deg[v]) + 0.1 * (1 / (1 + mean_neighbor_deg[v]))
    
    # This gives higher score to low-degree vertices and vertices with low-degree neighbors.
    # But we want vertices with HIGH-degree neighbors?
    # Because if neighbors have high degree, they are less likely to be picked, so v is a "safe" pick.
    
    # So:
    # score[v] = 1 / (1 + deg[v]) + 0.1 * (mean_neighbor_deg[v] / (1 + mean_neighbor_deg[v]))
    
    # Let's try this.
    
    # But we need to be careful with the greedy algorithm.
    # The greedy algorithm picks the highest score first.
    # So we want the "best" vertices to have the highest score.
    
    # Let's try a few variations and see which one makes sense.
    
    # Variation 1:
    # score1 = -deg + 0.5 * mean_neighbor_deg
    
    # Variation 2:
    # score2 = 1 / (1 + deg) + 0.1 * mean_neighbor_deg / (1 + mean_neighbor_deg)
    
    # Variation 3:
    # score3 = -deg + 0.1 * sum(deg[neighbors])
    
    # Let's try Variation 1 first.
    
    # But wait, the directive says "STRUCTURAL change: a different design or modelling family".
    # The current program uses "Lowest degree first".
    # Variation 1 is still based on degree.
    # Let's try something more structural.
    
    # How about using the "clustering coefficient" or "triangles"?
    # A vertex in many triangles is "dense" and might be less good for independent set?
    # Or more good?
    # If v is in many triangles, then its neighbors are connected to each other.
    # So if we pick v, we block all its neighbors, but they are already "competing" with each other.
    # So picking v might be more efficient?
    
    # Let's compute the number of triangles for each vertex.
    # This is O(n^3) in worst case, but n=200, so 8e6 operations, which is fine.
    
    # Actually, we can compute it using matrix multiplication.
    # Number of triangles containing v = (A^3)[v,v] / 2 ? No.
    # Number of triangles in the graph = trace(A^3) / 6.
    # Number of triangles containing v = (A^2 @ A)[v,v] / 2 ?
    # Actually, (A^2)[u,v] is the number of common neighbors of u and v.
    # So the number of triangles containing v is sum_{u in neighbors(v)} (A^2)[u,v] / 2?
    # No, each triangle is counted twice?
    
    # Let's compute it directly.
    
    triangles = np.zeros(n, dtype=np.int64)
    for i in range(n):
        neighbors_i = np.where(adj[i])[0]
        if len(neighbors_i) > 0:
            # Count common neighbors for each pair in neighbors_i
            # This is O(deg(i)^2)
            for j in range(len(neighbors_i)):
                for k in range(j+1, len(neighbors_i)):
                    u = neighbors_i[j]
                    v = neighbors_i[k]
                    if adj[u, v]:
                        triangles[i] += 1
    
    # So triangles[i] is the number of triangles containing vertex i.
    
    # Let's try:
    # score = -deg + 0.1 * triangles
    
    # Or:
    # score = -deg + 0.01 * triangles / (1 + deg)
    
    # Let's combine all features:
    # score = -deg + 0.5 * mean_neighbor_deg + 0.01 * triangles
    
    # But we need to be careful with the scales.
    # deg is up to 199.
    # mean_neighbor_deg is up to 199.
    # triangles is up to C(199, 2) ~ 20000.
    
    # So we need to normalize or use small weights.
    
    # Let's try:
    # score = -deg + 0.5 * mean_neighbor_deg + 0.001 * triangles
    
    # But this is just a heuristic.
    
    # Let's try a different approach:
    # Use the "greedy independent set" size as a feature?
    # No, that's what we're trying to optimize.
    
    # Let's try a machine learning approach? No, we don't have training data.
    
    # Let's try a spectral approach.
    # The eigenvectors of the adjacency matrix might give a good ordering.
    # The principal eigenvector gives a "centrality" score.
    # Vertices with high centrality are "important" and might be good to pick?
    # Or bad to pick?
    
    # Let's compute the principal eigenvector.
    
    # But the directive says "STRUCTURAL change".
    # Let's try the core decomposition idea from the inspiration.
    
    # "assigning exponentially decaying weights to vertices based on their removal order in the core peeling process"
    
    # So, we compute the removal order from the core peeling.
    # Then, score[v] = exp(-lambda * removal_order[v])
    
    # But we need to define lambda.
    # And we need to compute the removal order correctly.
    
    # Let's implement the core peeling correctly.
    
    # We'll use the heap-based peeling.
    # But we need to get the core numbers, not just the removal order.
    
    # Actually, the removal order in the peeling process is such that vertices with lower core numbers are removed first.
    # So, if we assign score[v] = exp(-lambda * removal_order[v]), then vertices with high core numbers (removed late) get high scores.
    
    # But do we want to pick vertices with high core numbers?
    # Vertices with high core numbers are in dense subgraphs.
    # In dense subgraphs, the independent set is small.
    # So maybe we want to pick vertices with LOW core numbers?
    
    # So, score[v] = exp(-lambda * (n - removal_order[v]))?
    # Or score[v] = -removal_order[v]?
    
    # Let's try:
    # score[v] = -removal_order[v]
    
    # This means vertices removed last (high core numbers) have the highest scores.
    
    # But wait, the current program uses "Lowest degree first", which is similar to "Lowest core number first".
    # So this might be similar to the current program.
    
    # Let's try the opposite:
    # score[v] = removal_order[v]
    
    # This means vertices removed first (low core numbers) have the highest scores.
    
    # But the current program already does something similar.
    
    # Let's try a combination:
    # score[v] = -deg[v] + 0.1 * removal_order[v]
    
    # This combines degree and core peeling order.
    
    # Let's implement the core peeling correctly.
    
    # We'll use the heap-based algorithm.
    
    # But we need to be careful with the heap updates.
    
    # Let's implement it.
    
    # Actually, for n=200, we can just use a simple O(n^2) peeling.
    
    # active = [True] * n
    # curr_deg = deg.copy()
    # removal_order = [0] * n
    # step = 0
    # while True:
    #   find active vertex with min curr_deg
    #   if none, break
    #   remove it, set removal_order[v] = step
    #   step += 1
    #   for each neighbor u of v that is active:
    #     curr_deg[u] -= 1
    
    # This is O(n^2) which is fine.
    
    active = [True] * n
    curr_deg = deg.copy()
    removal_order = [0] * n
    step = 0
    
    while step < n:
        # Find active vertex with min curr_deg
        min_deg = float('inf')
        min_v = -1
        for i in range(n):
            if active[i] and curr_deg[i] < min_deg:
                min_deg = curr_deg[i]
                min_v = i
        if min_v == -1:
            break
        active[min_v] = False
        removal_order[min_v] = step
        step += 1
        # Update neighbors
        for u in np.where(adj[min_v])[0]:
            if active[u]:
                curr_deg[u] -= 1
    
    # Now, let's try a few scoring functions.
    
    # Option 1: score = -removal_order
    # Option 2: score = removal_order
    # Option 3: score = -deg + 0.1 * removal_order
    # Option 4: score = -deg + 0.5 * mean_neighbor_deg
    
    # Let's try Option 3.
    
    # But we need to normalize removal_order to [0, 1] or something?
    # No, the greedy algorithm just sorts by score.
    
    # Let's try:
    # score = -deg + 0.1 * removal_order
    
    # But removal_order is up to 199, and deg is up to 199.
    # So 0.1 * removal_order is up to 19.9, which is small compared to deg.
    # So this is mostly -deg.
    
    # Let's try:
    # score = -deg + removal_order
    
    # This gives equal weight to degree and removal order.
    
    # But we need to be careful: removal_order is 0 for the first removed vertex, and n-1 for the last.
    # So a vertex removed last has a high score bonus.
    
    # Let's try this.
    
    score = -deg + np.array(removal_order, dtype=float)
    
    # But we need to return a numpy array.
    
    return score

# EVOLVE-BLOCK-END
