import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute all triples (i, j, k) with i < j < k
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    # Convert triples to numpy array for vectorized operations
    # T is shape (1140, 3)
    T = np.array(triples, dtype=np.int64)
    
    def is_valid(adj_np):
        """
        Check if graph is K3,3-free.
        adj_np: numpy array of shape (n,), where adj_np[i] is a bitmask of neighbors of i.
        Returns True if no triple has >= 3 common neighbors.
        """
        # For each triple (i, j, k), compute common neighbors: adj[i] & adj[j] & adj[k]
        # We want to check if any of these has bit_count >= 3.
        
        # Extract the three adjacency masks for each triple
        # T[:, 0] are the i's, T[:, 1] are the j's, T[:, 2] are the k's
        a_i = adj_np[T[:, 0]]
        a_j = adj_np[T[:, 1]]
        a_k = adj_np[T[:, 2]]
        
        common = a_i & a_j & a_k
        
        # Count bits in each common mask
        # np.bit_count is available in numpy 1.17+? Actually, standard numpy doesn't have bit_count on arrays directly in older versions, 
        # but we can use a lookup table or rely on Python's int.bit_count if we convert.
        # However, for speed, let's try to use a vectorized popcount.
        # Since n=20, the masks fit in 32-bit integers.
        
        # A simple way: convert to Python ints and use bit_count? That's slow.
        # Better: use a lookup table for 8-bit chunks? Or just use the fact that if bit_count >= 3, it's invalid.
        # Let's check if any common mask has >= 3 bits set.
        
        # We can check: if common & (common - 1) & (common - 2) ... no.
        # Let's just convert to a list of ints and check? 1140 items is small.
        # But we want speed.
        
        # Alternative: Use a precomputed table for 20-bit masks? 2^20 is 1M entries. Too big to precompute quickly?
        # Actually, we can just loop over the 1140 triples in Python, it's fast enough.
        # Let's benchmark mentally: 1140 iterations * (3 list accesses + 2 ANDs + 1 bit_count) is very fast (<1ms).
        
        for c in common:
            # c is a numpy int64
            if c.bit_count() >= 3:
                return False
        return True

    def count_edges(adj_np):
        # Sum bit counts of all adjacency masks, divide by 2
        total = 0
        for mask in adj_np:
            total += mask.bit_count()
        return total // 2

    def adj_to_edges(adj_np):
        edges = []
        for u in range(n):
            v = adj_np[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Initial graph: Start with a random dense graph and prune?
    # Or start with Paley?
    # Let's try a random start with p=0.5 and prune, then anneal.
    
    # To ensure we have a valid start, we can start with an empty graph and add edges?
    # Or start with a random graph and remove violating edges.
    
    def make_valid(adj_np):
        """Remove edges until valid. Returns True if successful within limits."""
        attempts = 0
        while True:
            # Find a violation
            # Re-use the logic from is_valid but find the specific triple
            T_np = T
            a_i = adj_np[T_np[:, 0]]
            a_j = adj_np[T_np[:, 1]]
            a_k = adj_np[T_np[:, 2]]
            common = a_i & a_j & a_k
            
            viol_idx = -1
            for idx, c in enumerate(common):
                if c.bit_count() >= 3:
                    viol_idx = idx
                    break
            
            if viol_idx == -1:
                return True
            
            i, j, k = T_np[viol_idx]
            common_mask = common[viol_idx]
            
            # Pick a random edge from the common neighbors to remove.
            # We need to remove an edge (x, y) such that removing it breaks the K3,3.
            # The K3,3 is formed by {i,j,k} and 3 common neighbors.
            # Removing any edge between {i,j,k} and the common neighbors might not break it if there are other common neighbors.
            # Actually, we just need to reduce the common neighbor count of this triple to < 3.
            # So we should remove an edge (u, v) where u in {i,j,k} and v in common_mask.
            
            candidates = []
            for u in [i, j, k]:
                # Neighbors of u in common_mask
                neigh_in_common = adj_np[u] & common_mask
                while neigh_in_common:
                    low = neigh_in_common & (-neigh_in_common)
                    v = low.bit_length() - 1
                    candidates.append((u, v))
                    neigh_in_common ^= low
            
            if not candidates:
                # Should not happen if common_mask is not 0
                return False
            
            u, v = random.choice(candidates)
            adj_np[u] &= ~(1 << v)
            adj_np[v] &= ~(1 << u)
            
            attempts += 1
            if attempts > 1000:
                return False

    best_adj = None
    best_count = 0
    best_edges = []
    
    def update_best(adj_np):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj_np)
        if c > best_count:
            best_count = c
            best_adj = adj_np.copy()
            best_edges = adj_to_edges(adj_np)

    random.seed(42)
    
    # Simulated Annealing
    # State: adjacency masks
    # Move: Add or Remove a random edge
    # Energy: -number of edges (we want to maximize edges, so minimize -edges)
    # Constraint: Must be valid.
    # If a move makes it invalid, we reject it (hard constraint).
    
    # Initial state
    # Start with a graph that is likely to be valid but dense.
    # Let's try starting with a random graph p=0.4 and pruning.
    
    adj = np.zeros(n, dtype=np.int64)
    for u, v in all_pairs:
        if random.random() < 0.4:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    if not make_valid(adj):
        # Fallback to empty
        adj = np.zeros(n, dtype=np.int64)
        
    update_best(adj)
    
    T_current = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    current_edges = count_edges(adj)
    
    while time.time() < deadline - 5:
        # Pick a random edge to toggle
        u, v = random.choice(all_pairs)
        
        # Try to add edge
        if not (adj[u] & (1 << v)):
            # Add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                new_edges = current_edges + 1
                # Always accept if better
                current_edges = new_edges
                update_best(adj)
            else:
                # Reject
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        else:
            # Try to remove edge (to allow future additions)
            # Only remove if we have "heat" to allow worse solutions
            # Or if it helps? Removing an edge never helps directly, but might unlock.
            # In SA, we accept worse moves with probability exp(-delta/T)
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Removing an edge always keeps the graph valid (subgraph of valid graph is valid? 
            # Wait. If G is K3,3-free, is G-e K3,3-free? Yes. Removing edges cannot create a K3,3.
            # So the new state is always valid.
            
            new_edges = current_edges - 1
            delta = new_edges - current_edges # -1
            
            # Accept with probability exp(-delta/T) = exp(1/T) ? No.
            # Energy E = -edges.
            # E_new = -(E_old - 1) = -E_old + 1.
            # Delta E = 1.
            # Prob = exp(-Delta E / T) = exp(-1/T).
            
            if random.random() < np.exp(-1.0 / T_current):
                current_edges = new_edges
            else:
                # Reject
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Cool down
        T_current *= cooling_rate
        if T_current < T_min:
            T_current = T_min

    # Final local search: Greedy add
    improved = True
    while improved:
        improved = False
        for u, v in all_pairs:
            if adj[u] & (1 << v):
                continue
            old_u, old_v = adj[u], adj[v]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_valid(adj):
                improved = True
                break
            else:
                adj[u] = old_u
                adj[v] = old_v
        
    update_best(adj)
    
    if not best_edges:
        return [[i, i + 1] for i in range(n - 1)]
        
    return best_edges

