import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Spectral Partitioning with Periphery Boosting
    1. Convert adjacency to a float matrix.
    2. Use Power Iteration to find the principal eigenvector (Fiedler-like vector for connected components).
       This vector separates the graph into high-centrality (core) and low-centrality (periphery) vertices.
    3. The greedy decoder picks high-score vertices first.
       We want the decoder to pick the periphery first, because periphery vertices have fewer neighbors,
       thus blocking fewer other vertices. This leaves more room for subsequent picks.
       The core vertices have many neighbors; picking one blocks many others. We want to pick these late
       or not at all if blocked by periphery picks.
    4. Score = - (centrality) + noise. Lower centrality -> Higher score.
    5. Refine: We can also consider the "local independence" potential.
       A vertex is a good candidate if it has many neighbors that are NOT in the current candidate set.
       However, since we only return static scores, we rely on the spectral separation.
       
    To improve upon simple degree:
    - Power iteration on A gives the eigenvector corresponding to the largest eigenvalue.
    - In a graph with a clear core-periphery structure, this eigenvector will have high values for core nodes and low for periphery.
    - We invert this: Score = -Eigenvector.
    
    Let's also try a "2-step" refinement:
    - Compute degree.
    - Compute "neighborhood density" or similar?
    
    Actually, let's look at the inspirations. They used local search on permutations.
    The current program uses local search on scores.
    
    A structural change: **Iterative Peeling Simulation as a Scorer**
    Instead of optimizing the order via search, we can *simulate* the greedy process to estimate the "value" of each vertex being picked early.
    
    Alternative Structural Change: **Randomized Restarts with Best-First Selection**
    The current program does local search.
    Let's try a **Multi-Start Greedy with Voting**.
    1. Generate K random permutations (or perturbations of degree-based order).
    2. Run the greedy decoder on each.
    3. For each vertex, count how often it was selected in the top M selections of these runs?
       No, we need a single priority list.
       
    Let's try **Simulated Annealing on the Order** but with a much faster evaluation using bitsets.
    The current program's bottleneck is likely the `get_is_size` function which loops over N vertices and updates a boolean array.
    For N=200, this is 200 * 200 = 40k ops per eval.
    If we do 1000 evals, that's 40M ops. In Python, this is slow (~40s).
    
    We can speed up the evaluation significantly by using **Bitsets** (Python integers).
    N=200 fits in a 256-bit integer (or 4 64-bit words, but Python ints are arbitrary precision).
    
    New Strategy: **Bitset-Accelerated Simulated Annealing on Permutation**
    1. Represent the "selected" set and "blocked" set as integers.
    2. Precompute the neighbor mask for each vertex as an integer.
    3. Evaluation of a permutation:
       - Iterate through vertices in order.
       - If vertex is not blocked:
         - Select it (count++)
         - Block it and its neighbors: `blocked_mask |= (1 << v) | neighbor_mask[v]`
    4. This evaluation is O(N) big-int operations, which are very fast in Python (C implementation).
    5. Use this fast evaluator to run a robust Simulated Annealing or Hill Climbing on the permutation.
    
    This is a structural change in *implementation* (bitsets vs arrays) enabling a different *search* (more aggressive SA) within the time limit.
    The previous program likely timed out or did few iterations due to slow evaluation.
    
    Let's implement this.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor masks as Python integers
    # Bit i is 1 if vertex i is a neighbor
    neighbor_masks = [0] * n
    for i in range(n):
        mask = 0
        # adj[i] is a boolean vector
        # We need to convert this to an integer where bit j is set if adj[i][j] is True
        # np.nonzero gives indices
        idxs = np.nonzero(adj[i])[0]
        for j in idxs:
            mask |= (1 << j)
        neighbor_masks[i] = mask
    
    # Helper to evaluate a permutation (list of vertex indices in order)
    def eval_perm(perm):
        blocked = 0
        count = 0
        for v in perm:
            bit_v = 1 << v
            if not (blocked & bit_v):
                count += 1
                # Block v and its neighbors
                blocked |= bit_v | neighbor_masks[v]
        return count

    # Initial permutation: Lowest Degree First (LDF)
    deg = adj.sum(axis=1)
    # Sort by degree ascending. Ties broken randomly.
    # We need a list of indices sorted by (degree, random)
    # To allow random tie-breaking, we can create a list of tuples (deg, random, idx)
    # But for reproducibility of the "structure", let's just sort by degree.
    # To avoid being stuck, we will perturb this.
    
    # Create initial permutation
    indices = np.arange(n)
    # Shuffle first to break ties randomly
    np.random.shuffle(indices)
    # Stable sort by degree
    # np.argsort is stable? No, default is quicksort (not stable).
    # We can use np.lexsort or just sort a list of tuples.
    # For n=200, sorting a list of tuples is fine.
    init_order = sorted(range(n), key=lambda x: (deg[x], np.random.rand()))
    current_perm = init_order
    current_score = eval_perm(current_perm)
    
    best_perm = current_perm[:]
    best_score = current_score
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    while (time.time() - start_time) < deadline:
        if no_improve > max_no_improve:
            break
            
        # Generate neighbor permutation: Swap two random vertices
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
            
        # Swap
        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Evaluate
        new_score = eval_perm(current_perm)
        
        # Delta
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
            current_score = new_score
            no_improve = 0
            
            if new_score > best_score:
                best_score = new_score
                best_perm = current_perm[:]
        else:
            # Revert
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            no_improve += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    # Convert best_perm to scores
    # We need scores such that argsort(-scores) == best_perm
    # So score[best_perm[k]] should be high for small k.
    # Let score[best_perm[k]] = n - k
    scores = np.zeros(n)
    for k, v in enumerate(best_perm):
        scores[v] = n - k
        
    return scores
