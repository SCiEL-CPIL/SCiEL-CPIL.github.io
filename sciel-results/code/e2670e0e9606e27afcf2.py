"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex. The greedy decoder picks vertices
    in descending order of score, skipping neighbors of already-picked vertices.
    
    Strategy:
    1. Compute a base score based on "independence potential":
       - Low degree is good (fewer neighbors to block).
       - High "clustering" of neighbors is bad (neighbors block each other? No, neighbors block us).
       - Actually, we want to pick vertices that don't share neighbors with other high-priority vertices?
       
    Let's use a simple iterative local improvement on the order.
    The static heuristic "lowest degree" is decent.
    A stronger heuristic is "Minimum Sum of Neighbor Degrees" or similar.
    
    However, the knowledge suggests: "Direct local search optimization of the greedy decoder outcome via 2-opt swaps allows the algorithm to correct specific local suboptimalities... Measured: 1.038 -> 1.135."
    
    So, I should implement a local search on the priority scores (or the order) to maximize the independent set size produced by the greedy decoder.
    
    Steps:
    1. Generate initial order (e.g., by degree, or random).
    2. Evaluate the greedy independent set size.
    3. Try to improve the order via swaps (2-opt) or moves.
    4. Return the scores corresponding to the final order.
    
    Constraints:
    - 200 nodes.
    - 5 graphs.
    - 120s limit.
    - Must return scores (floats) such that descending sort gives the order.
    
    Let's implement a simple hill-climbing on the permutation of vertices.
    Since n=200, the permutation space is huge, but we can do local swaps.
    
    Algorithm:
    1. Compute degrees.
    2. Initial order: sort by degree (ascending).
    3. Compute independent set size for this order.
    4. For a number of iterations (or until time limit):
       - Pick a random pair (i, j) or a local neighborhood.
       - Swap their positions in the order.
       - Recompute independent set size (can be done efficiently if we track the set, but for simplicity and given n=200, a full recompute is O(n^2) ~ 40k ops, which is fast. 1000 swaps * 40k = 40M ops, well within 120s).
       - If size increases, keep the swap.
    
    Wait, the function `priority` returns scores. The evaluator sorts by scores.
    So I need to return scores that reflect the final order.
    If the final order is `perm`, I can return scores `n - i` for vertex `perm[i]`.
    
    Let's refine the recompute step.
    Greedy:
    available = set(all vertices)
    picked = []
    for v in order:
        if v in available:
            picked.append(v)
            available.remove(v)
            for u in neighbors[v]:
                available.discard(u)
    
    This is O(N + E) per evaluation.
    N=200, E~1000 (for ER graph with p=0.5, E~10000? No, 200*199/2 * 0.5 ~ 10000 edges. E~10k).
    So one evaluation is ~10k ops.
    10,000 evaluations = 100M ops. In Python, this might be slow (100M ops ~ 10-30s).
    We have 120s. So we can do maybe 1000-5000 swaps.
    
    To speed up, we can use numpy or bitsets.
    Bitset: N=200 fits in 4 64-bit integers or a single Python int (arbitrary precision).
    Python ints are efficient for bit operations.
    
    Let's use Python ints for bitsets.
    adj_bits[i] = bitmask of neighbors of i.
    
    Greedy with bitsets:
    available_mask = (1 << N) - 1
    picked_count = 0
    for v in order:
        bit = 1 << v
        if available_mask & bit:
            picked_count += 1
            # Remove v and its neighbors from available
            available_mask &= ~(bit | adj_bits[v])
    
    This is very fast.
    
    Local Search:
    Start with order sorted by degree.
    Iterate:
      - Try swapping random pairs? Or adjacent pairs?
      - Adjacent pairs (2-opt) are local and effective for ordering.
      - Or random pairs to escape local minima.
    
    Let's try a mix: mostly adjacent swaps, some random.
    
    Time management:
    Check time every 100 swaps.
    
    """
    N = adjacency.shape[0]
    # Convert to bitmasks
    adj_bits = [0] * N
    # adjacency is a bool matrix
    # We can use numpy to get neighbors, then convert to int
    for i in range(N):
        mask = 0
        # neighbors of i
        # adjacency[i] is a bool array
        # find indices where True
        # To be fast, we can just loop j
        # But N=200, so N^2 = 40000 checks per graph.
        # 5 graphs * 40000 = 200k checks. Negligible.
        for j in range(N):
            if adjacency[i, j]:
                mask |= (1 << j)
        adj_bits[i] = mask
        
    # Initial order: lowest degree first
    degrees = [adj_bits[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    def get_is_size(order):
        available_mask = (1 << N) - 1
        count = 0
        for v in order:
            bit = 1 << v
            if available_mask & bit:
                count += 1
                available_mask &= ~(bit | adj_bits[v])
        return count
    
    current_size = get_is_size(order)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Local search
    # We will try to improve by swapping elements.
    # Since N=200, we can't do all pairs.
    # Random sampling of pairs.
    
    # To be deterministic and robust, let's do a fixed number of iterations if time allows,
    # or until time limit.
    
    # A better heuristic for initial order might be "MinSum" (sum of neighbor degrees)
    # Let's stick to degree for now, local search should fix it.
    
    # We can also try to move a vertex to a better position.
    # But swaps are simpler.
    
    # Let's use a simple random restart or just hill climbing on swaps.
    # Given the time, hill climbing on random swaps is good.
    
    # To make it more effective, we can try all adjacent swaps first (one pass), then random.
    
    # Pass 1: Adjacent swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            # Swap i and i+1
            order[i], order[i+1] = order[i+1], order[i]
            new_size = get_is_size(order)
            if new_size > current_size:
                current_size = new_size
                improved = True
            else:
                # Revert
                order[i], order[i+1] = order[i+1], order[i]
    
    # Pass 2: Random swaps
    # Use a fixed seed for reproducibility? No, the evaluator uses fixed graphs,
    # but we don't know the seed. Randomness is fine.
    # However, if we want to be safe, we can use a deterministic pseudo-random sequence
    # based on the graph structure? No, just use Python's random.
    
    import random
    # To ensure we don't get stuck in a bad local minimum, we can do multiple passes
    # or random swaps.
    
    # Let's do random swaps for the remaining time.
    while time.time() < deadline - 5:
        # Pick two random indices
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        
        order[i], order[j] = order[j], order[i]
        new_size = get_is_size(order)
        if new_size > current_size:
            current_size = new_size
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            
    # Convert order to scores
    # The vertex at position 0 in order should have the highest score.
    # Score = N - index
    scores = np.zeros(N, dtype=float)
    for idx, v in enumerate(order):
        scores[v] = N - idx
        
    return scores

# EVOLVE-BLOCK-END
