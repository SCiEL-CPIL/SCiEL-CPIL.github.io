import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1)
    
    # Compute Core Numbers (k-core decomposition)
    # This is an O(m) algorithm
    def compute_core_numbers(adj, degs):
        n = len(degs)
        core = np.zeros(n, dtype=int)
        # Use a bucket sort approach for efficiency
        # Max degree is n-1
        max_deg = int(np.max(degs)) if n > 0 else 0
        buckets = [[] for _ in range(max_deg + 1)]
        for i in range(n):
            buckets[int(degs[i])].append(i)
        
        # Process vertices in order of increasing degree
        # We need to update degrees of neighbors when a vertex is removed
        remaining_deg = degs.copy()
        removed = np.zeros(n, dtype=bool)
        
        # This simple implementation might be O(n^2) in worst case if not careful, 
        # but for n=200 it's fine.
        # Actually, let's use a standard peeling algorithm.
        
        # Since n is small (200), we can use a simpler O(n^2) or O(n*m) method
        # or just use the standard peeling with a heap.
        
        import heapq
        heap = [(int(degs[i]), i) for i in range(n)]
        heapq.heapify(heap)
        
        while heap:
            d, v = heapq.heappop(heap)
            if removed[v]:
                continue
            # The core number of v is at least d? No, it's the current degree in the remaining graph.
            # Actually, the core number is the max k such that v is in the k-core.
            # Peeling: remove vertex with min degree. Its core number is its current degree.
            core[v] = d
            removed[v] = True
            
            # Update neighbors
            for u in neighbors[v]:
                if not removed[u]:
                    # Decrease degree of u
                    # We need to update the heap
                    # This lazy deletion approach works
                    remaining_deg[u] -= 1
                    heapq.heappush(heap, (int(remaining_deg[u]), u))
        
        return core

    core_numbers = compute_core_numbers(adjacency, degrees)
    
    # Fast greedy MIS evaluation using NumPy for speed
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if neighbors[v].size > 0:
                    selected[neighbors[v]] = True
        return count
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_order = None
    
    def update_best(order, size):
        nonlocal best_size, best_order
        if size > best_size:
            best_size = size
            best_order = order[:]
    
    # Generate initial orderings
    initial_orders = []
    
    # 1. Min Degree
    initial_orders.append(np.argsort(degrees))
    
    # 2. Min Core Number
    initial_orders.append(np.argsort(core_numbers))
    
    # 3. Min (Core Number, Degree) - Lexicographic sort
    # Combine into a single score for sorting
    # Core number is more important? Or Degree?
    # Let's try Core Number first, then Degree
    score_cd = core_numbers * n + degrees
    initial_orders.append(np.argsort(score_cd))
    
    # 4. Min (Degree, Core Number)
    score_dc = degrees * n + core_numbers
    initial_orders.append(np.argsort(score_dc))
    
    # 5. Random
    initial_orders.append(rng.permutation(n))
    
    # 6. Noisy Min Degree
    noisy_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_deg))
    
    # 7. Noisy Min Core
    noisy_core = core_numbers + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_core))
    
    # 8. Max "Independence Potential" (Inverse Degree)
    # Vertices with low degree have high potential
    potential = 1.0 / (degrees + 1.0)
    initial_orders.append(np.argsort(-potential))
    
    # 9. Dynamic Independent Potential (layer-based scoring)
    # Simulate greedy removal: pick vertex with max "independent potential",
    # remove it and neighbors, assign layer scores
    def compute_dynamic_potential(adj, degs, nbrs):
        n = len(degs)
        remaining_deg = degs.copy().astype(float)
        removed = np.zeros(n, dtype=bool)
        layer_scores = np.zeros(n)
        current_layer = 0
        
        while not np.all(removed):
            # Find vertices with max "independent potential" among remaining
            # Potential = 1 / (remaining_degree + 1)
            pot = np.where(removed, -np.inf, 1.0 / (remaining_deg + 1.0))
            # Pick the one with max potential (break ties by lower degree)
            max_pot = np.max(pot)
            candidates = np.where((pot == max_pot) & (~removed))[0]
            if len(candidates) == 0:
                break
            # Among candidates, pick the one with lowest remaining degree
            min_deg = np.min(remaining_deg[candidates])
            final_candidates = candidates[remaining_deg[candidates] == min_deg]
            v = final_candidates[0]
            
            # Assign layer score
            layer_scores[v] = current_layer + 1
            removed[v] = True
            
            # Remove neighbors
            for u in nbrs[v]:
                if not removed[u]:
                    removed[u] = True
                    remaining_deg[u] = 0
                    layer_scores[u] = current_layer + 1
            
            # Update remaining degrees
            for u in nbrs[v]:
                if not removed[u]:
                    remaining_deg[u] -= 1
            
            current_layer += 1
        
        return layer_scores
    
    dyn_pot = compute_dynamic_potential(adjacency, degrees, neighbors)
    initial_orders.append(np.argsort(dyn_pot))
    
    # 10. Combined: Core number + Dynamic potential
    combined_score = core_numbers.astype(float) + dyn_pot
    initial_orders.append(np.argsort(combined_score))
    
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Local Search: 2-Opt Hill Climbing
    # This acts as a high-precision polish pass, deterministically improving the order.
    
    # Sort initial orders by their greedy size to start from the best
    evaluated = [(greedy_mis_size(o), o) for o in initial_orders]
    evaluated.sort(key=lambda x: -x[0])
    
    # Pick top 4 distinct-ish initial orders for local search
    ls_candidates = []
    seen_hashes = set()
    for size, order in evaluated:
        # Simple hash: tuple of first 10 elements
        h = tuple(order[:10])
        if h not in seen_hashes:
            ls_candidates.append(order)
            seen_hashes.add(h)
        if len(ls_candidates) >= 4:
            break
            
    if not ls_candidates:
        ls_candidates = [initial_orders[0]]
        
    for start_order in ls_candidates:
        if time.time() > deadline - 15:
            break
            
        order = start_order.copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        # 2-Opt Local Search with early termination and adaptive pass count
        max_passes = 15
        for _pass in range(max_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            # Check swaps in a smart order: prioritize swaps involving
            # vertices that were recently selected or have high impact
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n):
                    # Try swapping i and j
                    order[i], order[j] = order[j], order[i]
                    new_size = greedy_mis_size(order)
                    if new_size > current_size:
                        current_size = new_size
                        update_best(order, current_size)
                        improved = True
                    else:
                        # Revert swap
                        order[i], order[j] = order[j], order[i]
            
            if not improved:
                break
                    
    # Convert best_order to scores
    if best_order is None:
        scores = np.zeros(n)
        order = np.argsort(degrees)
        for rank, v in enumerate(order):
            scores[v] = n - rank
        return scores
        
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
