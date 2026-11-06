"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: 
    1. Convert graph to bitmask representation for O(1) neighbor updates.
    2. Initialize ordering using a Min-Degree greedy heuristic (strong baseline for MIS).
    3. Optimize the ordering using Simulated Annealing with Swap and Insertion moves.
    4. Return scores corresponding to the best ordering found.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    
    # Precompute neighbor masks for fast simulation
    # adj_mask[i] is an integer where bit j is 1 if i and j are connected
    adj_mask = [0] * n
    for i in range(n):
        neighbors = np.where(adj_bool[i])[0]
        mask = 0
        for j in neighbors:
            mask |= (1 << j)
        adj_mask[i] = mask
        
    # Full mask of all vertices
    full_mask = (1 << n) - 1
    
    def simulate_greedy(order):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        available = full_mask
        selected_count = 0
        
        for v in order:
            # Check if v is still available
            if (available & (1 << v)) == 0:
                continue
            
            # Select v
            selected_count += 1
            
            # Remove v and its neighbors from available set
            # ~(1 << v) removes v
            # ~adj_mask[v] removes neighbors
            available &= ~(1 << v)
            available &= ~adj_mask[v]
            
            # Early termination if no vertices left
            if available == 0:
                break
                
        return selected_count

    # --- Initialization: Min-Degree Heuristic ---
    # This heuristic repeatedly picks the vertex with the minimum degree in the remaining graph.
    # This produces a good initial ordering for MIS.
    
    def min_degree_ordering():
        remaining = set(range(n))
        order = []
        # We need to efficiently find min degree in the remaining subgraph.
        # Since n=200, we can just iterate.
        # deg[v] in the remaining graph = number of neighbors of v that are in 'remaining'
        
        # Precompute neighbor sets for faster lookup
        neighbor_sets = [set(np.where(adj_bool[i])[0]) for i in range(n)]
        
        while remaining:
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                # Count neighbors in remaining
                deg = len(neighbor_sets[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            order.append(min_v)
            remaining.remove(min_v)
            
        return order

    initial_order = min_degree_ordering()
    current_order = initial_order
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # --- Simulated Annealing ---
    T = 10.0  # Initial temperature
    cooling = 0.9995
    
    # We will alternate between Swap and Insertion moves
    # Insertion move: pick i, j, move element at i to position j
    
    def apply_swap(order, i, j):
        order[i], order[j] = order[j], order[i]
        
    def apply_insertion(order, i, j):
        # Move element at index i to index j
        # If i < j, elements i+1..j shift left
        # If i > j, elements j..i-1 shift right
        elem = order[i]
        del order[i]
        order.insert(j, elem)

    steps = 0
    while time.time() < deadline:
        steps += 1
        if steps % 10000 == 0:
            T *= cooling
            
        # Generate a move
        move_type = random.random()
        if move_type < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            new_order = current_order.copy()
            apply_swap(new_order, i, j)
        else:
            # Insertion
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            new_order = current_order.copy()
            apply_insertion(new_order, i, j)
            
        new_size = simulate_greedy(new_order)
        
        delta = new_size - current_size
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            current_order = new_order
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
                
        # If we have a lot of time, we can also do a local improvement pass occasionally
        # But SA should handle it.
        
    # Convert best_order to scores
    # Higher score = selected earlier
    # best_order[0] is the first selected, so it should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
# EVOLVE-BLOCK-END
