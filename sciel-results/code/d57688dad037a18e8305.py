"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def priority(adjacency):
    """
    Assigns a score to each vertex such that a greedy decoder (descending score,
    skipping neighbours of picked vertices) yields a large independent set.
    
    Strategy:
    1. Convert adjacency to a set of neighbor sets for O(1) lookups.
    2. Simulate the greedy process multiple times with different tie-breaking
       strategies or random perturbations to estimate the marginal contribution
       of each vertex to the independent set size.
    3. Specifically, we use a 'randomized greedy' estimator. For each trial, we
       pick vertices in a random order (or priority order with noise) and track
       how many times each vertex is selected.
    4. The final score is proportional to the frequency of selection.
    
    This is a robust heuristic because it directly optimizes for the metric
    (independent set size) under the specific greedy decoder structure,
    accounting for the fact that selecting a high-degree vertex early might
    block many others, whereas selecting low-degree vertices might allow more
    total vertices.
    
    We also incorporate a static component based on degree to ensure
    determinism and speed, but the core improvement comes from the
    simulation-based estimation of 'independence value'.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = adjacency.shape[0]
    
    # Convert to list of sets for fast neighbor checks
    # adjacency is a bool matrix
    neighbors = [set(np.where(adjacency[i, :])[0]) for i in range(n)]
    
    # Precompute degrees
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=float)
    
    # We will run multiple simulations.
    # In each simulation, we simulate the greedy selection.
    # To make it fast, we use a priority queue or sorted list.
    # However, for 200 nodes, even a simple O(n^2) per simulation is fine.
    # 200^2 = 40,000 ops. 1000 simulations = 40M ops. This is very fast in Python if optimized.
    
    # Let's use a randomized priority approach.
    # For each vertex, we assign a random priority.
    # Then we sort by priority.
    # We simulate the greedy selection.
    # We count how many times each vertex is selected.
    
    # Number of simulations:
    # Let's aim for as many as possible within the time budget.
    # We'll start with 100 and increase if fast.
    
    counts = np.zeros(n, dtype=int)
    num_sims = 0
    
    # We can also use a deterministic base score based on degree.
    # Low degree is generally better for greedy MIS.
    # But let's let the simulation decide.
    
    # Optimization: Use numpy for the selection process?
    # The greedy step is: pick v, remove v and all neighbors from available set.
    # This is inherently sequential.
    
    # Let's try a few different strategies in parallel (conceptually).
    # Strategy 1: Random order.
    # Strategy 2: Degree-based order (lowest degree first).
    # Strategy 3: Randomized degree-based.
    
    # We'll mix them.
    
    # Precompute a list of vertices
    vertices = list(range(n))
    
    # We'll run simulations.
    # To speed up, we can use a bitset or boolean array for available vertices.
    # But with n=200, a set is likely faster in Python due to O(1) removal.
    
    # Let's benchmark a single simulation.
    # 1. Shuffle vertices (or sort by key).
    # 2. Iterate, if available, select, remove neighbors.
    
    # We'll use a helper function to simulate one run.
    def simulate_order(order):
        available = set(vertices)
        selected = []
        for v in order:
            if v in available:
                selected.append(v)
                available.remove(v)
                # Remove neighbors
                for u in neighbors[v]:
                    if u in available:
                        available.remove(u)
        return selected
    
    # We want to estimate the "value" of each vertex.
    # If a vertex is selected frequently across random orders, it's "robustly independent".
    # However, the decoder is fixed: it picks by descending score.
    # So we want a score vector S such that the greedy selection by S is good.
    # This is a fixed-point problem.
    
    # Alternative approach:
    # The greedy decoder is deterministic given scores.
    # We can try to find scores that maximize the size.
    # This is hard to optimize directly.
    
    # Heuristic:
    # Vertices that are "central" in the independence sense (i.e., their exclusion doesn't hurt much, or they are easy to include) should have high scores.
    # Actually, we want to include as many as possible.
    # If vertex v is selected, it blocks its neighbors.
    # So we prefer vertices with few neighbors (low degree).
    # But also, if v is a neighbor of many high-degree vertices, maybe we should pick v first to block them? No, we want to pick v, so we want v to be available.
    
    # Let's look at the "independence number" of the neighborhood.
    # If N(v) has a large independent set, maybe v is less critical?
    
    # Let's stick to the simulation-based frequency.
    # If we run random greedy simulations, the frequency of selection is a proxy for how "greedy-friendly" the vertex is.
    # If we set the score to this frequency, the greedy decoder will pick the most frequently selected vertices first.
    # This tends to converge to a good solution.
    
    # To improve, we can use a weighted combination of:
    # 1. Negative degree (static).
    # 2. Simulation frequency (dynamic).
    
    # Let's run simulations.
    # We'll use random permutations.
    
    # Optimization: 
    # Instead of full simulation, we can just track the first few picks?
    # No, the whole set matters.
    
    # Let's run as many simulations as possible.
    # We'll check time every 100 simulations.
    
    # Initialize counts
    # We'll also track the total size of independent sets found to normalize?
    # No, we just want the order.
    
    # Let's use a smarter simulation:
    # Instead of random order, we can use an order based on current degrees (dynamic greedy).
    # But the decoder uses a FIXED score.
    # So the simulation must use the FIXED score to be a true estimate.
    # This is circular.
    
    # Break the cycle:
    # 1. Start with static scores (e.g., -degree).
    # 2. Simulate greedy with these scores.
    # 3. Update scores based on simulation results (e.g., vertices selected early get higher scores?).
    # 4. Repeat.
    
    # This is like a self-consistent field iteration.
    
    scores = -degrees.copy()
    
    # Iteration loop
    for _ in range(50):  # Max 50 iterations
        if time.time() > deadline:
            break
        
        # Sort vertices by current scores (descending)
        order = np.argsort(-scores)
        
        # Simulate greedy with this order
        selected = simulate_order(order)
        
        # Update scores:
        # Vertices in the selected set should have high scores.
        # Vertices not selected should have low scores.
        # But we need to preserve the relative order that led to this selection.
        
        # A simple update:
        # score[v] += 1 if v in selected else -1
        # This is like a reinforcement learning approach.
        
        # However, this might just reinforce the initial degree-based order.
        
        # Better update:
        # The score should reflect the "regret" of not picking a vertex.
        
        # Let's try a different heuristic:
        # Compute the "independent set potential" of each vertex.
        # Potential(v) = 1 + sum(Potential(u) for u in N(v) if u is not v)
        # This is recursive.
        
        # Let's try: Score(v) = -degree(v) + lambda * (number of neighbors that are "bad" to pick)
        
        # Actually, let's look at the complement graph.
        # MIS in G = Max Clique in G_complement.
        # Greedy on G with score S is equivalent to Greedy on G_comp with score S?
        # No, the decoder skips neighbors in G. In G_comp, it skips non-neighbors.
        # So it's not directly equivalent.
        
        # Let's stick to the frequency of selection in random simulations.
        # It is a robust measure of "independence".
        
        # We'll run random simulations in parallel with the fixed-score simulation.
        
        # Let's just run random simulations and accumulate counts.
        # Then use counts as the score.
        
        pass # We will change the strategy below
    
    # Revised Strategy:
    # 1. Compute a static score: -degree.
    # 2. Run K random greedy simulations.
    # 3. Count selections.
    # 4. Final score = alpha * (-degree) + beta * (counts).
    
    # This combines the structural prior (degree) with the empirical performance.
    
    # Let's implement this.
    
    counts = np.zeros(n, dtype=float)
    total_size = 0.0
    num_sims = 0
    
    # We will run simulations until time runs out or a fixed number.
    # Let's aim for 1000 simulations.
    
    # To speed up simulation, we can use a list of available vertices and a set for O(1) removal.
    # But we need to iterate in a specific order.
    
    # For random simulations:
    # 1. Generate random permutation.
    # 2. Simulate.
    
    # We can vectorize the "is available" check? No, it's sequential.
    
    # Let's just loop.
    
    # We'll use a numpy random permutation for speed.
    
    # We'll run a batch of simulations.
    while time.time() < deadline:
        # Batch size
        batch_size = 10
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            # Random order
            order = np.random.permutation(n)
            
            # Simulate
            available = set(vertices)
            selected_count = 0
            for v in order:
                if v in available:
                    selected_count += 1
                    counts[v] += 1
                    available.remove(v)
                    for u in neighbors[v]:
                        if u in available:
                            available.remove(u)
            
            total_size += selected_count
            num_sims += 1
        
        # Check time
        if time.time() > deadline:
            break
    
    # If we didn't run any simulations (unlikely), fallback to degree.
    if num_sims == 0:
        return -degrees
    
    # Combine scores
    # Normalize counts
    max_count = counts.max()
    if max_count > 0:
        normalized_counts = counts / max_count
    else:
        normalized_counts = counts
    
    # Normalize degrees
    max_deg = degrees.max()
    if max_deg > 0:
        normalized_degrees = degrees / max_deg
    else:
        normalized_degrees = degrees
    
    # Weighted combination
    # We want high counts and low degrees.
    # Score = w1 * normalized_counts - w2 * normalized_degrees
    
    # Let's try equal weights first.
    scores = 0.5 * normalized_counts - 0.5 * normalized_degrees
    
    return scores
# EVOLVE-BLOCK-END
