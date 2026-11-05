import numpy as np
import time
import random

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Minimum Degree Peeling Construction
    1. Instead of optimizing a score vector, we directly construct a high-quality
       independent set using the Minimum Degree heuristic.
    2. The heuristic iteratively selects the vertex with the lowest degree in the
       remaining graph, adds it to the independent set, and removes it along with
       its neighbors.
    3. We assign scores based on the order of selection: vertices selected earlier
       (in the peeling process) get higher scores. Vertices removed as neighbors
       get low scores.
    4. This ensures the greedy decoder, which picks in descending score order,
       exactly replicates the Minimum Degree peeling process.
    5. We use multiple restarts with randomized tie-breaking to find a better
       peeling order.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as lists for fast access
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def simulate_min_degree(noise=None, seed=0):
        """
        Performs Minimum Degree Peeling.
        Returns the order of selection (list of vertex indices).
        """
        if noise is not None:
            rng = np.random.default_rng(seed)
            # Add small noise to degrees to break ties randomly
            # This helps escape local optima in the peeling order
            pass 
        else:
            rng = None

        # active[i] is True if vertex i is still in the graph
        active = np.ones(n, dtype=bool)
        # deg[i] is the current degree of vertex i in the active graph
        deg = adj.sum(axis=1).astype(int).copy()
        
        selection_order = []
        
        # We need an efficient way to find the min-degree active vertex.
        # For n=200, a simple loop is O(n^2) which is 40,000 ops. Very fast.
        # We can just scan for the min degree among active vertices each time.
        
        for _ in range(n):
            # Find active vertex with minimum degree
            min_deg = float('inf')
            min_vertex = -1
            
            # If we have noise, we might want to add it to the comparison
            # But standard Min-Degree is deterministic. 
            # To allow restarts, we can randomize the tie-breaking.
            
            # Optimization: Only check active vertices
            # Using a simple loop is fine for n=200
            for i in range(n):
                if active[i]:
                    if deg[i] < min_deg:
                        min_deg = deg[i]
                        min_vertex = i
                    elif deg[i] == min_deg:
                        # Random tie-breaking to allow different paths
                        if rng is not None and rng.random() < 0.5:
                            min_vertex = i
                            
            if min_vertex == -1:
                break
                
            selection_order.append(min_vertex)
            
            # Remove min_vertex and its neighbors
            active[min_vertex] = False
            deg[min_vertex] = 0 # Not strictly necessary but good hygiene
            
            for nb in neighbors_list[min_vertex]:
                if active[nb]:
                    active[nb] = False
                    deg[nb] = 0 # It's removed, so degree doesn't matter for selection
                    # Important: We must update the degrees of the *other* neighbors of nb?
                    # No, the graph is being pruned.
                    # When we remove a vertex u, the degree of all its active neighbors v decreases by 1.
                    # My current logic above does NOT update the degrees of the neighbors of the removed neighbors.
                    # It only marks them as inactive.
                    # This is CORRECT for the "peeling" interpretation where we remove the closed neighborhood.
                    # However, the "degree" of a vertex v in the *remaining* graph should reflect
                    # the number of its *active* neighbors.
                    # If I just mark neighbors as inactive, I don't update the degrees of vertices
                    # that are neighbors to those removed vertices.
                    # Example: A-B-C. Remove A. B is removed. C's degree should drop by 1 (if B was active).
                    # In my loop: 
                    # 1. Select A (min deg).
                    # 2. Mark A inactive.
                    # 3. Mark B inactive (neighbor of A).
                    # 4. C is still active. But C's degree was 1 (neighbor B). Now B is inactive.
                    #    C's degree in the active graph is 0.
                    #    But my `deg[C]` is still 1.
                    #    This is a BUG in the simple implementation if I don't update degrees of surviving vertices.
                    
            # Corrected Logic for Degree Updates:
            # When we remove a set of vertices R, for every vertex v in R, 
            # we must decrement the degree of all active neighbors of v.
            
            # Let's re-implement the inner loop properly.
            # This is getting complex for a simple loop. 
            # Alternative: Since n=200, we can just recalculate degrees from scratch 
            # or maintain a bitset.
            # Actually, for n=200, O(n^2) per step is 200*200*200 = 8,000,000 ops. 
            # In Python, this might be slow. 8M ops * 200 steps? No, 200 steps total.
            # 200 steps * O(n^2) update = 200 * 40,000 = 8,000,000 operations.
            # Python can do ~10M simple ops/sec. So 1 second per run. 
            # With 10 restarts, that's 10 seconds. Acceptable.
            
            # Let's restart the function with a cleaner, correct implementation.
            return None # Placeholder to force rewrite

    def run_min_degree(seed=0):
        active = np.ones(n, dtype=bool)
        # We will maintain a list of active vertices to iterate over
        active_vertices = list(range(n))
        
        selection_order = []
        
        # Precompute adj as a set of sets for faster neighbor checks? 
        # neighbors_list is already there.
        
        # To correctly update degrees, we can just compute the degree of an active vertex
        # on the fly by counting active neighbors.
        # deg(v) = sum(1 for u in neighbors_list[v] if active[u])
        
        for _ in range(n):
            if not active_vertices:
                break
                
            min_deg = float('inf')
            min_vertex = -1
            rng = np.random.default_rng(seed + len(selection_order))
            
            # Find min degree vertex among active ones
            for v in active_vertices:
                # Calculate current degree
                # This is O(deg(v)). Average deg is small.
                d = 0
                for u in neighbors_list[v]:
                    if active[u]:
                        d += 1
                        if d >= min_deg: # Pruning
                            break
                            
                if d < min_deg:
                    min_deg = d
                    min_vertex = v
                    # Reset RNG for tie-breaking if needed, but let's just pick the first min for now
                    # or use RNG to pick among ties.
                    # For simplicity and speed, let's stick to deterministic min, 
                    # but use different seeds for the *initial* randomization of the graph?
                    # No, the graph is fixed.
                    # Let's add a small random weight to the degree to break ties.
                    # d_effective = d + rng.random() * 0.1
                    # But that changes the logic.
                    # Let's just use the first one found with min degree.
                    # To get variety, we can shuffle the active_vertices list before the loop?
                    # That would make it O(n log n) per step.
                    
            if min_vertex == -1:
                break
                
            selection_order.append(min_vertex)
            
            # Remove min_vertex and its neighbors
            active[min_vertex] = False
            removed_set = {min_vertex}
            
            for nb in neighbors_list[min_vertex]:
                if active[nb]:
                    active[nb] = False
                    removed_set.add(nb)
                    
            # Update active_vertices list
            # Remove all vertices in removed_set
            # List removal is O(n). 
            # active_vertices = [v for v in active_vertices if v not in removed_set]
            # This is O(n^2) total if done naively.
            # Since n=200, it's fine.
            
            removed_set_tuple = tuple(removed_set)
            active_vertices = [v for v in active_vertices if v not in removed_set]
            
        return selection_order

    # This O(n^2) per step implementation might be too slow in Python if not optimized.
    # 200 steps * (200 vertices * 200 neighbors check) = 8,000,000 checks.
    # Each check is a Python loop. 8M iterations in Python takes ~0.5-1s.
    # With 10 restarts, 5-10s. This is safe.
    
    # Let's optimize the inner degree calculation using NumPy if possible, 
    # but the "active" mask changes.
    # Actually, we can precompute the adjacency matrix as a sparse structure or just use the bool matrix.
    
    # Let's try a different approach for speed: 
    # Use the adjacency matrix `adj` (bool).
    # deg(v) = np.sum(adj[v, active])
    # This is a vectorized operation. O(n) per vertex.
    # Finding min: np.argmin(degrees[active])
    
    def run_min_degree_fast(seed=0):
        active = np.ones(n, dtype=bool)
        selection_order = []
        rng = np.random.default_rng(seed)
        
        for _ in range(n):
            if not np.any(active):
                break
            
            # Calculate degrees of all active vertices
            # adj @ active gives the degree of each vertex in the active subgraph
            # adj is (n,n) bool. active is (n,) bool.
            # Result is (n,) int.
            degrees = adj @ active
            
            # Mask out inactive vertices by setting their degree to inf
            # Or just find min among active indices
            active_indices = np.where(active)[0]
            
            if len(active_indices) == 0:
                break
                
            # Get degrees for active vertices
            active_degrees = degrees[active_indices]
            
            # Add small noise to break ties randomly
            # This ensures different restarts can explore different paths
            noise = rng.random(len(active_indices)) * 0.01
            effective_degrees = active_degrees + noise
            
            # Find index of min effective degree
            min_idx_in_active = np.argmin(effective_degrees)
            min_vertex = active_indices[min_idx_in_active]
            
            selection_order.append(min_vertex)
            
            # Remove min_vertex and its neighbors
            active[min_vertex] = False
            neighbors = np.where(adj[min_vertex])[0]
            active[neighbors] = False
            
        return selection_order

    def score_from_order(order):
        scores = np.full(n, -1.0)
        # Higher score = picked earlier
        # The greedy decoder picks in descending score order.
        # So the first vertex in `order` (picked first in MIS construction) should have the highest score.
        for i, v in enumerate(order):
            scores[v] = float(len(order) - i)
        return scores

    def simulate_scores(scores):
        """
        Simulate the greedy decoder to verify the score.
        """
        order = np.argsort(-scores, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = np.where(adj[v])[0]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    best_scores = None
    best_val = 0
    
    # Run multiple restarts of Min-Degree with different noise seeds
    max_time = 115.0
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time - 5:
            break
            
        order = run_min_degree_fast(seed=r)
        scores = score_from_order(order)
        val = simulate_scores(scores)
        
        if val > best_val:
            best_val = val
            best_scores = scores
            
        # Early stop if we reach a perfect or near-perfect bound?
        # For ER(200, 0.5), max MIS is roughly n/2 * something.
        # If we hit a high value, we can stop.
        
    if best_scores is None:
        # Fallback: random scores
        best_scores = np.random.rand(n)
        
    return best_scores
