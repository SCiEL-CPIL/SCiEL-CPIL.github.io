import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Constructive Greedy with Local Refinement.
    1. Initialize an order using a strong heuristic (degree + neighbor density).
    2. Perform a "Local Search" on the independent set itself:
       - Identify vertices in the independent set (IS).
       - Identify vertices NOT in the IS but not blocked by IS (i.e., could be added if we remove blockers).
       - Actually, since the decoder is fixed (greedy on order), we optimize the ORDER.
       - However, random swaps are inefficient. 
       - Alternative: Use a "Priority Queue" based construction.
       - We simulate the greedy process. At each step, instead of just picking the next in a static list, 
         we could theoretically choose the best vertex. But the decoder is fixed.
         So we must output a static list.
         
    Let's rethink the "Different Strategy":
    Current: Simulated Annealing on Permutation (Random Swaps).
    New: **Beam Search / A* on the State Space of Partial Orders?** 
    Too complex for 120s.
    
    How about: **Iterative Deepening Local Search (IDLS) on the Order?**
    Or better: **Greedy Construction with "Lookahead" Scoring.**
    
    Let's try a fundamentally different optimization: **Genetic Algorithm (GA) on Permutations**.
    Crossover: Order-based crossover.
    Mutation: Inversion or Swap.
    Selection: Tournament.
    This explores the space more broadly than SA, potentially escaping local minima better.
    
    Actually, a very effective strategy for MIS is **Degree-based Greedy with Dynamic Re-scoring** isn't allowed because the decoder is fixed.
    
    Let's go with **Tabu Search** on the permutation.
    - Move: Swap two vertices.
    - Tabu List: Keep a list of recent swaps to prevent cycling.
    - Aspiration Criterion: Accept tabu move if it improves best-known solution.
    - This is a different meta-heuristic family from SA.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbors
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = adj.sum(axis=1)
    
    def evaluate_order(order):
        """
        Evaluate the greedy independent set size for a given permutation.
        Returns: count
        """
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                # Block neighbors
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count

    # Initial Ordering: Sort by degree ascending (standard good heuristic for MIS)
    # But let's try a slightly more sophisticated initialization:
    # Score = degree + 0.1 * (sum of degrees of neighbors)
    # Low degree is generally better for being selected first.
    neighbor_deg_sum = np.array([np.sum(deg[neighbors_list[i]]) for i in range(n)])
    init_score = deg + 0.1 * neighbor_deg_sum
    current_order = np.argsort(init_score, kind='stable')
    
    current_count = evaluate_order(current_order)
    best_order = current_order.copy()
    best_count = current_count
    
    rng = np.random.default_rng(42)
    
    # Tabu Search Parameters
    tabu_tenure = 10  # Keep last 10 moves in tabu list
    tabu_list = []    # List of (v1, v2, iteration)
    max_tabu_size = 100
    
    # We will perform a series of local searches.
    # Instead of random swaps, we evaluate all possible swaps involving the "frontier"
    # or just do random swaps but with Tabu memory.
    
    # To make it faster, we can limit the search to "promising" swaps.
    # A swap is promising if it involves a vertex that is currently selected and a vertex that is blocked,
    # or vice versa.
    
    # Let's stick to random swaps with Tabu, but add a "Restart" mechanism if stuck.
    
    iter_count = 0
    no_improve_count = 0
    restart_threshold = 2000
    
    while time.time() - start_time < deadline:
        iter_count += 1
        
        # Pick two random positions to swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        v_i = current_order[i]
        v_j = current_order[j]
        
        # Check Tabu
        is_tabu = False
        for (tv1, tv2, t_iter) in tabu_list:
            if (tv1 == v_i and tv2 == v_j) or (tv1 == v_j and tv2 == v_i):
                if iter_count - t_iter < tabu_tenure:
                    is_tabu = True
                    break
        
        # Perform Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_count = evaluate_order(current_order)
        
        # Aspiration: If it improves best, accept even if tabu
        if new_count > best_count:
            # Accept
            pass
        elif is_tabu and new_count <= best_count:
            # Reject: Swap back
            current_order[i], current_order[j] = current_order[j], current_order[i]
            continue
        else:
            # Accept if better or equal to current (hill climbing)
            # To avoid getting stuck, we allow some worse moves? 
            # Tabu Search usually does strict improvement or diversification.
            # Let's do: Accept if new_count >= current_count - 1 (allow slight deterioration)
            # But we must track current_count properly.
            if new_count < current_count - 1:
                # Reject if too bad
                current_order[i], current_order[j] = current_order[j], current_order[i]
                continue
            # If we reject, we should not update tabu list with the move we are undoing?
            # Standard TS: The move is made, evaluated, and if rejected, undone.
            # The tabu list prevents the *reverse* move immediately.
            
        # Update Current State
        current_count = new_count
        
        # Update Tabu List
        tabu_list.append((v_i, v_j, iter_count))
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)
            
        # Update Best
        if new_count > best_count:
            best_count = new_count
            best_order = current_order.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Diversification: If stuck for too long, restart with a perturbed order
        if no_improve_count > restart_threshold:
            # Perturb current best order
            best_order = best_order.copy()
            # Randomly swap 5 pairs
            for _ in range(5):
                p1 = rng.integers(0, n)
                p2 = rng.integers(0, n)
                best_order[p1], best_order[p2] = best_order[p2], best_order[p1]
            current_order = best_order.copy()
            current_count = evaluate_order(current_order)
            no_improve_count = 0
            tabu_list.clear() # Clear tabu on restart

    # Convert best_order to scores
    scores = np.zeros(n)
    for k, v in enumerate(best_order):
        scores[v] = n - k
        
    return scores
