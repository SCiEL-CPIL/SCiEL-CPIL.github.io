"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    n = adjacency.shape[0]
    adjacency = adjacency.astype(bool)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbor_lists], dtype=float)

    # Fast simulation: track 'available' vertices; a vertex is selected iff still available
    # when its turn comes.  O(n + |E_selected|) per call instead of O(n * avg_degree).
    def simulate_greedy(order):
        available = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if available[v]:
                count += 1
                available[v] = False
                nbrs = neighbor_lists[v]
                if len(nbrs) > 0:
                    available[nbrs] = False
        return count

    # K-core decomposition (min-core peeling): low core = peripheral
    def k_core_decomposition():
        core_number = np.zeros(n, dtype=int)
        removed = np.zeros(n, dtype=bool)
        current_degree = degrees.copy()
        changed = True
        while changed and time.time() < deadline - 60:
            changed = False
            for v in range(n):
                if not removed[v] and current_degree[v] < (core_number[v] + 1):
                    removed[v] = True
                    core_number[v] = int(current_degree[v])
                    changed = True
                    for nbr in neighbor_lists[v]:
                        if not removed[nbr]:
                            current_degree[nbr] -= 1
        for v in range(n):
            if not removed[v]:
                core_number[v] = int(current_degree[v])
        return core_number

    # Max-core (max-core peeling): high max_core = in densest subgraph
    def max_core_decomposition():
        max_core = np.zeros(n, dtype=int)
        removed = np.zeros(n, dtype=bool)
        current_degree = degrees.copy()
        changed = True
        while changed and time.time() < deadline - 60:
            changed = False
            for v in range(n):
                if not removed[v] and current_degree[v] > (max_core[v]):
                    # This vertex can't sustain a higher max-core
                    removed[v] = True
                    max_core[v] = int(current_degree[v])
                    changed = True
                    for nbr in neighbor_lists[v]:
                        if not removed[nbr]:
                            current_degree[nbr] -= 1
        for v in range(n):
            if not removed[v]:
                max_core[v] = int(current_degree[v])
        return max_core

    core_numbers = k_core_decomposition()
    max_core_numbers = max_core_decomposition()

    # Compute branching factor: how many vertices in the graph are "dependent" on this vertex
    # A vertex with high core number is more central; removing it affects more of the structure
    # We'll compute a "branching weight" based on core number and degree
    
    # Exponentially decaying weight based on removal order in core peeling
    # Vertices removed later (higher core) get higher priority to be selected first
    # But we also want to consider degree: among same core, lower degree might be better for independent set
    
    # Compute a combined structural score
    # High core number = structurally central = should be selected carefully
    # For maximum independent set, we want to select vertices that "block" fewer others
    # So: prefer vertices with low degree and low core number (peripheral, low degree)
    
    # But the k-core insight: vertices in high cores are part of dense subgraphs
    # In dense subgraphs, greedy selection is less efficient
    # So we should prioritize vertices in low cores (peripheral) that can be selected early
    
    # Score: lower is better for selection order (we'll negate for argsort)
    # Combine: core_number (lower is better for peripheral) + degree (lower is better)
    
    # Normalize features
    core_norm = core_numbers.astype(float) / (core_numbers.max() + 1e-9)
    max_core_norm = max_core_numbers.astype(float) / (max_core_numbers.max() + 1e-9)
    deg_norm = degrees / (degrees.max() + 1e-9)
    
    # Structural score: we want to select peripheral, low-degree vertices first
    structural_score = 0.6 * core_norm + 0.4 * deg_norm
    
    # Also compute a "branching factor" proxy: number of vertices that would be affected
    # if this vertex is selected (i.e., its degree in the remaining graph)
    # This is approximately the degree, but we can refine it
    
    # For now, use structural_score as initial priority
    # Lower structural_score = should be selected earlier
    
    # Generate initial orders
    rng = np.random.RandomState(42)
    
    initial_orders = []
    
    # 1. By structural score (ascending = select peripheral first)
    initial_orders.append(np.argsort(structural_score))
    
    # 2. By degree (ascending)
    initial_orders.append(np.argsort(degrees))
    
    # 3. By core number (ascending)
    initial_orders.append(np.argsort(core_numbers))
    
    # 4. Random
    initial_orders.append(rng.permutation(n))
    
    # 5. Degree with noise
    deg_noise = degrees + rng.uniform(0, 1, n) * 0.1
    initial_orders.append(np.argsort(deg_noise))
    
    # 6. Core number with noise
    core_noise = core_numbers.astype(float) + rng.uniform(0, 1, n) * 0.5
    initial_orders.append(np.argsort(core_noise))
    
    # 7. Combined with different weights
    alt_score = 0.3 * core_norm + 0.7 * deg_norm
    initial_orders.append(np.argsort(alt_score))
    
    # 8. Another random
    initial_orders.append(rng.permutation(n))
    
    # 9. Max-core (ascending = select from densest subgraphs first)
    initial_orders.append(np.argsort(max_core_numbers))
    
    # 10. Max-core + degree combined
    mc_deg = 0.5 * max_core_norm + 0.5 * deg_norm
    initial_orders.append(np.argsort(mc_deg))
    
    # 11. Core + max-core combined (peripheral in both senses)
    core_mc = 0.5 * core_norm + 0.5 * max_core_norm
    initial_orders.append(np.argsort(core_mc))
    
    # 12. Inverse degree (prefer high-degree? sometimes helps in sparse graphs)
    initial_orders.append(np.argsort(-degrees))
    
    # Local search to optimize the order
    best_order = initial_orders[0].copy()
    best_size = simulate_greedy(best_order)
    
    for idx, init_order in enumerate(initial_orders):
        if time.time() > deadline - 20:
            break
        
        curr_order = init_order.copy()
        curr_size = simulate_greedy(curr_order)
        
        # Phase 1: Adjacent swap local search
        improved = True
        max_passes = 10
        passes = 0
        while improved and time.time() < deadline - 10 and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                # Try swapping adjacent elements
                new_order = curr_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                new_size = simulate_greedy(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True
        
        # Phase 2: Random swap local search
        no_improve = 0
        max_no_improve = 300
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            new_order = curr_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_size = simulate_greedy(new_order)
            if new_size > curr_size:
                curr_size = new_size
                curr_order = new_order
                no_improve = 0
            else:
                no_improve += 1
        
        if curr_size > best_size:
            best_size = curr_size
            best_order = curr_order
    
    # Final polish: adjacent swap + insertion local search
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2 and passes < 12:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline - 2:
                break
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            new_size = simulate_greedy(new_order)
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
    
    # Insertion-based local search: try moving each vertex to a better position
    for _ in range(3):
        if time.time() > deadline - 2:
            break
        any_improve = False
        for i in range(n):
            if time.time() > deadline - 2:
                break
            v = best_order[i]
            # Try inserting v at each position j < i (move earlier)
            for j in range(max(0, i - 20), i):
                new_order = best_order.copy()
                # Remove v from position i
                new_order = np.delete(new_order, i)
                # Insert v at position j
                new_order = np.insert(new_order, j, v)
                new_size = simulate_greedy(new_order)
                if new_size > best_size:
                    best_size = new_size
                    best_order = new_order
                    any_improve = True
                    break
            if any_improve:
                break
        if not any_improve:
            break
    
    # Convert order to scores: earlier in order = higher score (more negative)
    scores = np.zeros(n)
    for pos, v in enumerate(best_order):
        scores[v] = -pos
    
    return scores


# EVOLVE-BLOCK-END
