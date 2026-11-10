# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import milp, Bounds, LinearConstraint

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph (edges connect pairs with forbidden difference).
    2. Use a hybrid approach:
       a. Quick MILP attempt (30s) for initial solution.
       b. Tabu search + local search for refinement.
       c. Multiple randomized greedy restarts with local search.
    3. The local search is optimized for speed using bitsets (integers) for adjacency.
    """
    m = 205
    k = 2
    
    # Step 1: Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency as bitmasks for fast local search
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute neighbor lists for faster iteration
    neighbors = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if (adj_mask[i] >> j) & 1:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    def is_valid(subset):
        if len(subset) == 0:
            return False
        if len(set(subset)) != len(subset):
            return False
        if any(x < 0 or x >= m for x in subset):
            return False
        s = set(subset)
        for a in subset:
            for b in s:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True

    best_set = []
    best_size = 0
    
    def update_best(subset):
        nonlocal best_set, best_size
        if is_valid(subset):
            if len(subset) > best_size:
                best_set = list(subset)
                best_size = len(subset)

    # --- Fast Local Search using bitmasks ---
    def local_search(initial_set, max_iter=2000, time_budget=5.0):
        """Fast hill-climbing local search."""
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        start_time = time.time()
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and (time.time() - start_time) < time_budget:
            improved = False
            iter_count += 1
            
            # Try 1-move: Add a node (most beneficial)
            added = False
            for v in range(m):
                if v not in current and (adj_mask[v] & current_mask) == 0:
                    current.add(v)
                    current_mask |= (1 << v)
                    improved = True
                    added = True
                    break
            
            # Try 1-for-2: Remove 1, add 2 (net gain +1)
            if not improved and len(current) >= 1:
                found = False
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    # Find candidates that can be added if u is removed
                    candidates = []
                    for v in range(m):
                        if v not in current and (adj_mask[v] & mask_without_u) == 0:
                            candidates.append(v)
                    # Try pairs from candidates
                    for i in range(len(candidates)):
                        v1 = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            v2 = candidates[j]
                            if not ((adj_mask[v1] >> v2) & 1):
                                current.discard(u)
                                current.add(v1)
                                current.add(v2)
                                current_mask ^= (1 << u)
                                current_mask |= (1 << v1)
                                current_mask |= (1 << v2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            # Try swap: Remove 1, add 1 (neutral, may enable further additions)
            if not improved:
                found = False
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    for v in range(m):
                        if v not in current and (adj_mask[v] & mask_without_u) == 0:
                            # Check if this swap enables an addition
                            new_mask = mask_without_u | (1 << v)
                            can_add = False
                            for w in range(m):
                                if w not in current and w != v and (adj_mask[w] & new_mask) == 0:
                                    can_add = True
                                    break
                            if can_add:
                                current.discard(u)
                                current.add(v)
                                current_mask ^= (1 << u)
                                current_mask |= (1 << v)
                                improved = True
                                found = True
                                break
                    if found:
                        break
            
            # Try 2-for-1: Remove 2, add 1 (net loss, but may be needed)
            if not improved and len(current) >= 2:
                found = False
                u_list = list(current)
                for i in range(len(u_list)):
                    u1 = u_list[i]
                    for j in range(i + 1, len(u_list)):
                        u2 = u_list[j]
                        mask_without = current_mask ^ (1 << u1) ^ (1 << u2)
                        for v in range(m):
                            if v not in current and (adj_mask[v] & mask_without) == 0:
                                current.discard(u1)
                                current.discard(u2)
                                current.add(v)
                                current_mask ^= (1 << u1)
                                current_mask ^= (1 << u2)
                                current_mask |= (1 << v)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

        return list(current)
    
    def tabu_search(initial_set, time_budget=15.0):
        """Tabu search to escape local optima."""
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        start_time = time.time()
        tabu_list = {}  # node -> iteration when it becomes non-tabu
        tabu_tenure = 8
        iter_count = 0
        best_set = set(current)
        best_size = len(current)
        no_improve = 0
        
        while (time.time() - start_time) < time_budget:
            iter_count += 1
            if no_improve > 200:
                break
            
            best_move = None
            best_score = -1000
            best_type = None
            best_data = None
            
            # Generate candidate moves
            # Type 1: Add a node
            for v in range(m):
                if v in current:
                    continue
                if tabu_list.get(v, 0) > iter_count:
                    continue
                if (adj_mask[v] & current_mask) == 0:
                    score = 1
                    if score > best_score:
                        best_score = score
                        best_move = v
                        best_type = 'add'
            
            # Type 2: Swap - remove u, add v
            for u in list(current):
                mask_without_u = current_mask ^ (1 << u)
                for v in range(m):
                    if v in current or v == u:
                        continue
                    if tabu_list.get(v, 0) > iter_count or tabu_list.get(u, 0) > iter_count:
                        continue
                    if (adj_mask[v] & mask_without_u) == 0:
                        # Count how many new nodes this enables
                        new_mask = mask_without_u | (1 << v)
                        enabled = 0
                        for w in range(m):
                            if w not in current and w != v and (adj_mask[w] & new_mask) == 0:
                                enabled += 1
                        score = enabled
                        if score > best_score:
                            best_score = score
                            best_move = (u, v)
                            best_type = 'swap'
            
            # Type 3: 1-for-2 - remove u, add v1, v2
            if len(current) >= 1:
                for u in list(current):
                    if tabu_list.get(u, 0) > iter_count:
                        continue
                    mask_without_u = current_mask ^ (1 << u)
                    candidates = []
                    for v in range(m):
                        if v in current:
                            continue
                        if tabu_list.get(v, 0) > iter_count:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            candidates.append(v)
                    for i in range(len(candidates)):
                        v1 = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            v2 = candidates[j]
                            if (adj_mask[v1] >> v2) & 1:
                                continue
                            score = 2
                            if score > best_score:
                                best_score = score
                                best_move = (u, v1, v2)
                                best_type = '1for2'
            
            # Execute best move
            if best_move is not None and best_score >= 0:
                if best_type == 'add':
                    v = best_move
                    current.add(v)
                    current_mask |= (1 << v)
                    tabu_list[v] = iter_count + tabu_tenure
                    
                elif best_type == 'swap':
                    u, v = best_move
                    current.discard(u)
                    current.add(v)
                    current_mask ^= (1 << u)
                    current_mask |= (1 << v)
                    tabu_list[v] = iter_count + tabu_tenure
                    tabu_list[u] = iter_count + tabu_tenure
                    
                elif best_type == '1for2':
                    u, v1, v2 = best_move
                    current.discard(u)
                    current.add(v1)
                    current.add(v2)
                    current_mask ^= (1 << u)
                    current_mask |= (1 << v1)
                    current_mask |= (1 << v2)
                    tabu_list[v1] = iter_count + tabu_tenure
                    tabu_list[v2] = iter_count + tabu_tenure
                    tabu_list[u] = iter_count + tabu_tenure
                
                # Track best
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            
            else:
                no_improve += 1
        
        # Final polish
        polished = local_search(list(best_set), max_iter=500, time_budget=2.0)
        return polished

    deadline = time.time() + 580.0
    
    # --- Quick MILP Attempt (30s max) ---
    try:
        n_vars = m
        c = -np.ones(n_vars)
        
        edges = []
        for i in range(m):
            for j in range(i + 1, m):
                if (adj_mask[i] >> j) & 1:
                    edges.append((i, j))
        
        n_constraints = len(edges)
        if n_constraints > 0:
            A = np.zeros((n_constraints, n_vars))
            for idx, (i, j) in enumerate(edges):
                A[idx, i] = 1
                A[idx, j] = 1
            
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n_vars)
            cons = LinearConstraint(A, ub=np.ones(n_constraints))
            
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                      options={"time_limit": 30, "mip_rel_gap": 0.01})
            
            if res.status in [0, 1] and res.x is not None:
                solution = res.x
                subset = [i for i in range(n_vars) if solution[i] > 0.5]
                if len(subset) > 0:
                    update_best(subset)
    except Exception:
        pass

    # --- Local Search Refinement of MILP result ---
    if best_size > 0:
        refined = local_search(best_set, max_iter=2000, time_budget=5.0)
        update_best(refined)
        
        # Tabu search
        if time.time() < deadline - 30:
            ts_result = tabu_search(best_set, time_budget=15.0)
            update_best(ts_result)
    
    # --- Randomized Greedy + Local Search + Tabu Search ---
    num_restarts = 0
    while time.time() < deadline - 20:
        num_restarts += 1
        
        # Random greedy construction
        order = list(range(m))
        random.shuffle(order)
        current = []
        current_mask = 0
        for candidate in order:
            if (adj_mask[candidate] & current_mask) == 0:
                current.append(candidate)
                current_mask |= (1 << candidate)
        
        # Local search
        refined = local_search(current, max_iter=1000, time_budget=3.0)
        update_best(refined)
        
        # Tabu search if promising
        if len(refined) >= best_size - 2 and time.time() < deadline - 25:
            ts_result = tabu_search(refined, time_budget=10.0)
            update_best(ts_result)
        
        # Safety: limit total restarts
        if num_restarts >= 50:
            break

    if best_set:
        return np.array(sorted(best_set), dtype=np.int64)
    
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
