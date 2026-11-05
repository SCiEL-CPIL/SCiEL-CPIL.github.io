"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
    return count

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses Simulated Annealing on the vertex permutation to optimize the greedy result.
    """
    deadline = time.time() + 115  # Safety margin before 120s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    degree = adjacency.sum(axis=1).astype(int)
    
    # Initial heuristic: Sort by (degree, index) to get a decent starting point
    # Lower degree is generally better for MIS greedy.
    init_order = np.argsort(degree, kind='stable')
    
    # State:
    # perm: list of vertices in selection order (index 0 is first picked)
    # pos: inverse permutation (pos[v] = index of v in perm)
    # selected: boolean array, True if vertex is in the independent set
    # blocked: boolean array, True if vertex is blocked by a selected neighbor
    
    perm = list(init_order)
    pos = [0] * n
    for i, v in enumerate(perm):
        pos[v] = i
        
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    
    # Initialize state
    current_count = 0
    for v in perm:
        if not blocked[v]:
            selected[v] = True
            current_count += 1
            # Mark neighbors as blocked
            neighbors = np.where(adjacency[v])[0]
            blocked[neighbors] = True
            
    best_count = current_count
    best_perm = perm.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Generate move: swap two positions in the permutation
        # Random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        vi = perm[i]
        vj = perm[j]
        
        # Evaluate delta
        delta = 0
        undo_ops = []
        
        # We need to simulate the effect of swapping vi and vj in the greedy process.
        # This is complex to do in O(1) generally, but we can approximate or do a local re-simulation
        # of the affected region.
        # However, a simpler and robust approach for SA is to just run the full greedy decode
        # if we do a full evaluation, but that is O(n^2) or O(n^2/64).
        # With n=200, n^2 = 40000 ops. 120s allows ~100M ops. So ~2500 full evaluations.
        # That's too few for SA.
        
        # Let's try a faster incremental update.
        # If we swap i and j, only the decisions for vertices between min(i,j) and max(i,j) might change?
        # No, because blocking is global. If a vertex u is blocked by v, and v moves, u's status might change.
        
        # Given the complexity of exact O(1) delta for greedy MIS with arbitrary swaps,
        # let's optimize the full evaluation using NumPy vectorization as much as possible.
        
        # Actually, let's try a different move: "2-opt" like, or just accept that we do
        # a fast full evaluation.
        
        # Let's implement a very fast greedy decoder in Python/NumPy.
        # The bottleneck is the loop.
        
        # Alternative Strategy:
        # Since full evaluation is expensive, let's use a "greedy hill climbing" with
        # random restarts or a simpler local search that is faster.
        
        # Let's stick to the SA idea but use a very fast decoder.
        # Decoder optimization:
        # Instead of iterating one by one, we can't easily vectorize the "skip if blocked" logic
        # because the blocking depends on previous picks.
        
        # However, we can precompute neighbor lists as lists of ints for faster access?
        # np.where(adjacency[v]) is okay.
        
        # Let's try to speed up the decoder itself.
        # Use a bitset? n=200 fits in 4 64-bit ints or 1 256-bit int.
        # Python ints are arbitrary precision, so we can use them as bitsets!
        
        # Bitset representation:
        # adj_bits[v] is an integer with bit u set if u is neighbor of v.
        
        if not hasattr(priority, '_adj_bits'):
             priority._adj_bits = None
        if priority._adj_bits is None or priority._adj_bits[0] is not adjacency:
             adj_bits = [0] * n
             for v in range(n):
                 bits = 0
                 neigh = np.where(adjacency[v])[0]
                 for u in neigh:
                     bits |= (1 << u)
                 adj_bits[v] = bits
             priority._adj_bits = (adjacency, adj_bits)
        else:
             adj_bits = priority._adj_bits[1]
             
        # Fast decoder using bitsets
        def decode_fast(perm_list):
            picked_mask = 0
            forbidden_mask = 0
            count = 0
            for v in perm_list:
                bit_v = 1 << v
                if forbidden_mask & bit_v:
                    continue
                # Pick v
                picked_mask |= bit_v
                count += 1
                # Add neighbors to forbidden
                forbidden_mask |= adj_bits[v]
            return count
            
        # Re-evaluate current state with fast decoder to ensure consistency?
        # No, we maintain state. But for SA, we need to evaluate the *new* permutation.
        # If we do a full decode for every move, it's O(n) Python loop iterations.
        # 200 iterations * ~50ns/iter? No, Python loop is ~100ns-1µs.
        # 200 * 1µs = 200µs per eval.
        # 120s / 200µs = 600,000 evals. This is feasible!
        
        # So we can do standard SA with full evaluation using the bitset decoder.
        
        # Swap in perm
        perm[i], perm[j] = perm[j], perm[i]
        
        new_count = decode_fast(perm)
        
        if new_count > current_count or (new_count == current_count and rng.random() < 0.5):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_perm = perm.copy()
                T = 0.5 * T # Restart cooling on improvement? Or just keep going.
        else:
            # Accept with probability exp(-delta/T)
            delta = current_count - new_count
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_count = new_count
                else:
                    # Revert
                    perm[i], perm[j] = perm[j], perm[i]
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert best_perm back to scores
    # The greedy decoder sorts by descending score.
    # So vertex at index 0 in best_perm should have the highest score.
    scores = np.zeros(n)
    for idx, v in enumerate(best_perm):
        scores[v] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
