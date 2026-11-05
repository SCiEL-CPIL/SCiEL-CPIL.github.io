import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Stationary Distribution Initialization + Hill Climbing
    1. Compute the stationary distribution of a random walk on the graph.
       - This identifies vertices that are "peripheral" (low visitation frequency).
       - In ER graphs, peripheral nodes are often leaves or in sparse regions, 
         which are good candidates for early selection in MIS.
    2. Initialize scores as -stationary_distribution (or similar negative weighting).
    3. Perform Hill Climbing:
       - Simulate the greedy decoder to get a baseline set size.
       - Iteratively try to swap the priorities of two vertices.
       - To make it efficient, we don't check all pairs. We check pairs that are 
         "adjacent" in the current ranking or randomly sampled pairs.
       - Accept a swap if it increases the independent set size.
       - Use a strict time limit to ensure we return before the 150s deadline.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Pre-compute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = np.array([len(neighbors_list[i]) for i in range(n)])
    
    # 1. Compute Stationary Distribution via Power Iteration
    # Random walk transition matrix P: P[i][j] = 1/deg(i) if adj[i][j] else 0
    # We want the stationary distribution pi such that pi = pi @ P
    # Since the graph might be disconnected, we add a small self-loop (teleportation) 
    # or just handle components. For ER graphs, usually connected.
    # Let's use a simple power iteration on the row-stochastic matrix.
    
    # Handle isolated nodes: if deg[i]==0, P[i][i]=1
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        d = deg[i]
        if d > 0:
            P[i, neighbors_list[i]] = 1.0 / d
        else:
            P[i, i] = 1.0
            
    # Initialize pi uniformly
    pi = np.ones(n, dtype=float) / n
    
    # Power iteration
    for _ in range(50):  # 50 iterations is usually enough for N=200
        pi_new = pi @ P
        # Normalize to prevent drift
        pi_new = pi_new / pi_new.sum()
        pi = pi_new
        
    # The stationary distribution gives the long-term probability of visiting a node.
    # Nodes with low pi are less "central" in the random walk sense.
    # We want to prioritize these nodes.
    # So score = -pi.
    # However, pure -pi might not be optimal. Let's mix it with -degree.
    # Actually, let's start with -pi and let local search fix it.
    
    scores = -pi.copy()
    
    # Helper to simulate greedy MIS size
    def simulate(s):
        # Get order of indices by descending score
        order = np.argsort(-s)
        active = set(range(n))
        count = 0
        for v in order:
            if v in active:
                count += 1
                active.remove(v)
                # Remove neighbors
                for u in neighbors_list[v]:
                    if u in active:
                        active.remove(u)
        return count

    # 2. Initial Evaluation
    best_val = simulate(scores)
    
    # 3. Hill Climbing Local Search
    # We will try to improve by swapping scores of pairs (i, j).
    # To be efficient, we prioritize pairs that are close in rank or random.
    # Given N=200, full pairwise is 20k pairs. Each sim is ~O(N^2) or less.
    # 20k * 40k ops = 800M ops. In Python, this is too slow for a full pass.
    # We need a faster simulation or fewer evaluations.
    
    # Optimization: Use a faster simulation?
    # The set-based simulation is okay for N=200.
    # Let's estimate: 1 sim ~ 0.5ms? 20k sims ~ 10s. 
    # We have ~140s. So we can do multiple passes or random sampling.
    
    rng = np.random.default_rng(42)
    no_improve_streak = 0
    max_no_improve = 500  # If we fail 500 swaps in a row, maybe restart or stop?
    
    # We'll use a mix of random swaps and "neighbor-in-rank" swaps.
    # But random is simpler and robust.
    
    while time.time() < deadline:
        # Pick two distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        new_val = simulate(scores)
        
        if new_val > best_val:
            best_val = new_val
            no_improve_streak = 0
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            no_improve_streak += 1
            
        # If stuck for a long time, we might be in a local optimum.
        # A simple kick: randomly perturb a few scores? 
        # Or just keep going, as random swaps can escape.
        # Given the time limit, we just keep going.
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return scores

# Fallback for safety
def priority(adjacency):
    start_time = time.time()
    deadline = 140.0
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = np.array([len(neighbors_list[i]) for i in range(n)])
    
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        d = deg[i]
        if d > 0:
            P[i, neighbors_list[i]] = 1.0 / d
        else:
            P[i, i] = 1.0
            
    pi = np.ones(n, dtype=float) / n
    for _ in range(50):
        pi = pi @ P
        pi = pi / pi.sum()
        
    scores = -pi.copy()
    
    def simulate(s):
        order = np.argsort(-s)
        active = set(range(n))
        count = 0
        for v in order:
            if v in active:
                count += 1
                active.remove(v)
                for u in neighbors_list[v]:
                    if u in active:
                        active.remove(u)
        return count

    best_val = simulate(scores)
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        scores[i], scores[j] = scores[j], scores[i]
        new_val = simulate(scores)
        
        if new_val > best_val:
            best_val = new_val
        else:
            scores[i], scores[j] = scores[j], scores[i]
            
        if time.time() > deadline - 5.0:
            break
            
    return scores
