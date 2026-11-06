"""Deterministic Multi-Start Local Search for Maximum Cut."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-start deterministic local search (Hill Climbing).
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute total edges for quick cut evaluation if needed, 
    # but we will use incremental updates for speed.
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """
        Compute the cut size for a given labeling.
        labels: 0/1 array of length n.
        """
        # s = 2*labels - 1
        s = 2.0 * labels - 1.0
        # Q = s @ adj @ s
        # Cut = (M - Q) / 2
        # M = total edges = adj.sum() / 2
        M = adj.sum() / 2.0
        Q = s @ adj @ s
        return (M - Q) / 2.0

    def local_search(labels):
        """
        Perform deterministic local search (1-flip hill climbing) until no improvement is found.
        Returns the final labels and the cut size.
        """
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        # s vector for fast gain calculation
        # gain of flipping v = s[v] * sum_u A[v,u] s[u]
        # Let's maintain s vector to avoid recomputing 2*labels-1 every time.
        s = 2.0 * current_labels - 1.0
        
        improved = True
        # To avoid infinite loops on ties (gain=0), we only accept if gain > 0.
        # If gain == 0, flipping doesn't change cut size. We can break ties randomly or just stop.
        # Stopping is safer for determinism and speed, but might get stuck in plateaus.
        # For Max Cut, plateaus are rare with random starts.
        
        while improved:
            improved = False
            # We can iterate in random order or fixed order. Random order helps escape symmetry.
            # But for speed, fixed order with vectorized gain check is faster.
            # Let's compute gains for ALL vertices at once.
            
            # neighbor_sum = adj @ s  (vector of length n)
            # gain[v] = s[v] * neighbor_sum[v]
            neighbor_sum = adj @ s
            gains = s * neighbor_sum
            
            # Find vertices with positive gain
            # We want to flip all vertices that improve the cut.
            # However, flipping multiple vertices simultaneously might interfere.
            # Standard local search flips one at a time.
            # Vectorized approach: Find all indices with gain > 0.
            # If we flip all of them at once, the gains were calculated based on the OLD state.
            # This is a "parallel" local search step. It is generally safe for 1-flip moves
            # as long as we don't flip adjacent vertices that cancel out? 
            # Actually, if u and v are adjacent, flipping both might not be equivalent to flipping one then the other.
            # Example: edge (u,v). 
            # If both are in group 0. s_u=1, s_v=1. 
            # gain_u = 1 * (sum A[u,k]s_k). Includes s_v=1.
            # gain_v = 1 * (sum A[v,k]s_k). Includes s_u=1.
            # If we flip both: s_u=-1, s_v=-1.
            # New gain for u would be different.
            # To be safe and simple, let's do sequential flips but vectorized selection?
            # No, sequential is safer. But n=100 is small.
            
            # Let's just iterate through vertices. 
            # Optimization: Only check vertices that have potential to improve.
            # But checking all is O(n^2) per sweep. n=100 -> 10k ops. Very fast.
            
            for v in range(n):
                # gain is already computed in gains[v]
                # But gains are stale if we flipped a neighbor earlier in this sweep.
                # So we must recompute gain for v after each flip? 
                # Or just recompute gains array every sweep?
                # Recomputing gains array is O(n^2). Doing it once per sweep is efficient.
                # But if we flip v, gains for neighbors of v change.
                
                # Strategy: 
                # 1. Compute all gains.
                # 2. Flip the vertex with the MAXIMUM gain.
                # 3. Repeat.
                # This is "Steepest Ascent" local search.
                
                # Let's implement Steepest Ascent.
                # It converges faster than First-Ascent (which stops at first improvement).
                
                # We already have gains from the start of the loop? No, we need fresh gains.
                # Let's restructure the inner loop.
                
                pass # Placeholder, logic moved below

        # Corrected Local Search Logic: Steepest Ascent
        # We will re-implement the loop inside the function to be clean.
        # The code above was just a sketch. Let's write the real one.
        
        return current_labels, current_cut

    # Real Local Search Implementation
    def local_search_sa(labels):
        """
        Steepest Ascent Local Search.
        """
        current_labels = labels.copy()
        s = 2.0 * current_labels - 1.0
        current_cut = compute_cut(current_labels)
        
        while True:
            # Compute gains for all vertices
            # gain[v] = change in cut if we flip v
            # gain[v] = s[v] * sum_u A[v,u] s[u]
            neighbor_sum = adj @ s
            gains = s * neighbor_sum
            
            # Find vertex with max gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 0:
                break # Local optimum reached
            
            # Flip the vertex
            v = max_gain_idx
            current_labels[v] = 1 - current_labels[v]
            s[v] = -s[v]
            current_cut += max_gain
            
            # Optional: Check time inside tight loop? 
            # n=100, convergence is very fast (usually < 100 flips).
            # So this loop is very fast.
            
        return current_labels, current_cut

    # Multi-Start Strategy
    # We have ~115 seconds.
    # Each local search takes < 1ms for n=100.
    # We can do thousands of restarts.
    
    start_time = time.time()
    num_restarts = 0
    
    # We will run until time limit.
    # Random restarts with different seeds.
    
    while time.time() < deadline:
        # Random initialization
        labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Run local search
        labels, cut = local_search_sa(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        num_restarts += 1
        
        # If we are very fast, we might want to do a second phase:
        # Perturb the best solution found so far and re-run local search.
        # This is a form of iterated local search (ILS).
        if num_restarts % 100 == 0:
            # Check time
            if time.time() > deadline:
                break
            
            # Perturb best_labels
            perturbed_labels = best_labels.copy()
            # Flip 1-5 random vertices
            k = np.random.randint(1, 6)
            flip_indices = np.random.choice(n, k, replace=False)
            perturbed_labels[flip_indices] = 1 - perturbed_labels[flip_indices]
            
            labels, cut = local_search_sa(perturbed_labels)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()

    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
