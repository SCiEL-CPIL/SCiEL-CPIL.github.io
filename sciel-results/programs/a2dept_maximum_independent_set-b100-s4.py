"""Combined strategy:
1. (From inspiration 1) Parameterized static scoring functions (degree, neighbor degree sum, coreness)
   explored via a grid search + random perturbations. This is fast and captures structural heuristics.
2. (From current program) Simulated Annealing on the vertex order to fine-tune the specific
   permutation, allowing for non-linear dependencies that static formulas might miss.
3. (From inspiration 3) Direct local search on the independent set (add/remove/swap) as a final
   polishing step, or as an alternative if SA fails to improve.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order (permutation).
    Returns: (count, selected_mask)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        # Mark neighbors as unavailable
        available[adjacency[v]] = False
        
    return count, selected

def _evaluate_scores(adjacency, scores):
    """
    Evaluate a set of scores by sorting them and running the greedy decoder.
    Returns: count
    """
    order = np.argsort(-scores, kind='stable')
    count, _ = _greedy_decode(adjacency, order)
    return count

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Efficient core number computation for small n
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        masked_degrees = np.where(remaining, deg, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = deg[v]
        
        remaining[v] = False
        neighbors = np.where(adjacency[v])[0]
        active_neighbors = neighbors[remaining[neighbors]]
        deg[active_neighbors] -= 1
        
    return core

def _search_best_scores(adjacency):
    """
    Search over parameterized scoring functions to find the best greedy outcome.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    
    # Normalize for better conditioning
    deg_norm = degrees / (degrees.max() + 1e-9)
    nds_norm = neighbor_deg_sum / (neighbor_deg_sum.max() + 1e-9)
    core_norm = coreness / (coreness.max() + 1e-9)
    
    best_count = -1
    best_scores = None
    
    # Grid search over alpha, beta, gamma
    # Score = -(alpha * deg + beta * nds + gamma * core)
    # We want low degree, low neighbor degree sum, low coreness to be high priority?
    # Wait, the decoder picks HIGH score first.
    # So we want the "good" vertices (low degree) to have HIGH score.
    # So score = -(alpha * deg_norm + ...)
    
    alphas = [0, 1, 2, 5]
    betas = [0, 0.5, 1, 2]
    gammas = [0, 0.5, 1]
    
    deadline = time.time() + 140.0
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                if time.time() > deadline:
                    return best_scores, best_count
                
                scores = -(alpha * deg_norm + beta * nds_norm + gamma * core_norm)
                count = _evaluate_scores(adjacency, scores)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
    
    # If no scores found (shouldn't happen), fallback to -degrees
    if best_scores is None:
        best_scores = -degrees
        best_count = _evaluate_scores(adjacency, best_scores)
        
    return best_scores, best_count

def _simulated_annealing(adjacency, initial_scores, deadline):
    """
    Refine the order using Simulated Annealing.
    """
    n = adjacency.shape[0]
    current_order = np.argsort(-initial_scores, kind='stable')
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    T = 1.0
    cooling_rate = 0.99
    max_steps = 10000
    
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            break
            
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_count, _ = _greedy_decode(adjacency, current_order)
        
        delta = new_count - current_count
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            if step > 2000:
                break
                
        step += 1
        
    # Convert best order to scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores, best_count

def _local_search_is(adjacency, initial_set, deadline):
    """
    Local search on the independent set itself.
    initial_set: boolean mask of vertices in the independent set.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = np.sum(current_set)
    
    best_set = current_set.copy()
    best_size = current_size
    
    # Precompute neighbors for speed
    # neighbors[v] is a boolean mask of neighbors of v
    neighbors = adjacency
    
    # Candidate vertices: not in set, and not neighbors of any vertex in set?
    # Actually, we can add a vertex if it's not in the set and not a neighbor of any vertex in the set.
    # A vertex u can be added if: not current_set[u] AND not any(current_set[neighbors[u]])
    
    # Let's maintain a "blocked" mask: vertices that are in the set OR are neighbors of the set.
    # blocked = current_set OR (sum over v in set of neighbors[v])
    # This is equivalent to the "closed neighborhood" of the set.
    
    def get_blocked_mask(s):
        if not np.any(s):
            return np.zeros(n, dtype=bool)
        # s @ adjacency gives sum of neighbors for each vertex? No.
        # If s is bool, s @ adjacency gives count of neighbors in s for each vertex.
        # A vertex v is blocked if s[v] is True OR count_neighbors_in_s[v] > 0.
        # Actually, if v is in s, it's blocked. If v has a neighbor in s, it's blocked.
        # So blocked[v] = s[v] OR (adjacency[v] @ s > 0)
        # Let's compute adjacency @ s (vector of length n, element i is sum of s[j] for j in neighbors of i)
        # Wait, adjacency is symmetric.
        # (adjacency @ s)[i] = sum_j A[i,j] s[j]. This is the number of neighbors of i that are in s.
        # So blocked[i] = s[i] OR ((adjacency @ s)[i] > 0)
        neighbor_counts = adjacency @ s
        return s | (neighbor_counts > 0)

    blocked = get_blocked_mask(current_set)
    
    while time.time() < deadline:
        # Find candidates: vertices not blocked
        candidates = np.where(~blocked)[0]
        
        if len(candidates) == 0:
            break
            
        # Try to add a candidate
        # Randomly pick a candidate
        v = candidates[np.random.randint(len(candidates))]
        
        # Check if adding v improves size (it always does by 1, but might block future additions)
        # For now, just add it if it's valid.
        # Actually, simple hill climbing: add a vertex, then try to remove one to gain 2?
        # Or just add if possible.
        
        # Add v
        current_set[v] = True
        current_size += 1
        
        # Update blocked
        # New blocked = old_blocked OR neighbors[v] OR {v}
        # But removing vertices from blocked is hard.
        # Let's just recompute blocked. It's O(n^2) which is fine for n=200.
        blocked = get_blocked_mask(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
            
        # Try to swap: remove a vertex from set, add two?
        # This is complex. Let's stick to simple addition for now, 
        # but simple addition will eventually block everything.
        # We need a mechanism to "undo" bad additions or try swaps.
        
        # Alternative: If we can't add, try swapping.
        # This local search is tricky. Let's rely more on SA.
        # But we can try a simple "kick" if stuck.
        # If len(candidates) == 0, we are stuck.
        # Let's remove a random vertex from the set to unblock some candidates.
        if len(candidates) == 0:
            # Remove a random vertex from the set
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) == 0:
                break
            v_to_remove = set_vertices[np.random.randint(len(set_vertices))]
            current_set[v_to_remove] = False
            current_size -= 1
            blocked = get_blocked_mask(current_set)
            
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # 1. Parameterized Search
    best_scores, best_count = _search_best_scores(adjacency)
    
    # 2. Simulated Annealing on the order derived from best_scores
    if time.time() < deadline - 10.0:
        sa_scores, sa_count = _simulated_annealing(adjacency, best_scores, deadline - 5.0)
        if sa_count > best_count:
            best_scores = sa_scores
            best_count = sa_count
    else:
        # If no time for SA, just use the grid search result
        pass
    
    # 3. Local Search on the Independent Set derived from best_scores
    # This is a bit redundant if SA was run, but might catch different local optima.
    # Let's try it if time permits.
    if time.time() < deadline - 5.0:
        order = np.argsort(-best_scores, kind='stable')
        _, init_set = _greedy_decode(adjacency, order)
        ls_set, ls_count = _local_search_is(adjacency, init_set, deadline - 1.0)
        
        if ls_count > best_count:
            # Convert the local search set to scores
            # Vertices in the set get high scores, others low.
            # To ensure the greedy decoder picks exactly this set (or a superset if possible, but greedy is fixed),
            # we assign scores such that the set vertices are picked first.
            # However, the greedy decoder might pick a DIFFERENT set if the scores are not carefully tuned.
            # If we set score[v] = 1 for v in ls_set and 0 for others, the greedy decoder will pick
            # all vertices in ls_set (since they are independent) and then try to pick others?
            # No, if v is in ls_set, it's picked. Its neighbors are blocked.
            # If we pick all of ls_set first, the remaining available vertices are those not in ls_set and not neighbors of ls_set.
            # The greedy decoder will then pick the highest score among those.
            # So we need to ensure that no vertex OUTSIDE ls_set has a higher score than those IN ls_set?
            # Actually, if we assign score 1 to ls_set and 0 to others, the decoder picks ls_set vertices first.
            # Then it picks any other available vertex with score 0.
            # This might ADD vertices to the set, which is good!
            # But it might pick a vertex that conflicts with a future better vertex?
            # Since we already found a local optimum ls_set, adding more vertices is only possible if they are independent of ls_set.
            # So the greedy decoder on scores (1 for ls_set, 0 for others) will produce a set that is a superset of ls_set?
            # Not necessarily. It picks ls_set vertices. Then it picks the best available non-neighbor.
            # This is effectively extending the independent set.
            # So the result size >= ls_count.
            
            scores = np.zeros(n)
            scores[ls_set] = 1.0
            best_scores = scores
            # We don't update best_count here because the actual decoder might do better (add more).
            # But for the return value, we just return the scores.
            
    return best_scores

# EVOLVE-BLOCK-END
