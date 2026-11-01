import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures:
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # 2. Greedy Phase to get an initial upper bound (LB)
    # This is fast and provides a good starting point for the solver.
    def greedy_solution(seed=0):
        random.seed(seed)
        gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 10.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run a few greedy passes to find a tight upper bound
    best_greedy_sol = None
    best_greedy_len = float('inf')
    for i in range(5):
        sol = greedy_solution(i)
        if sol is not None and len(sol) < best_greedy_len:
            best_greedy_len = len(sol)
            best_greedy_sol = sol
            
    if best_greedy_sol is None:
        # Fallback if greedy fails (shouldn't happen)
        return []

    # 3. Constraint Programming (CP-SAT)
    # We try to solve the exact problem using OR-Tools if available.
    # If OR-Tools is not available, we fall back to a more aggressive local search.
    
    try:
        from ortools.sat.python import cp_model
        solver_available = True
    except ImportError:
        solver_available = False

    if solver_available:
        model = cp_model.CpModel()
        
        # Decision variables: x_j = 1 if block j is selected
        x = [model.NewBoolVar(f"x_{j}") for j in range(num_k_blocks)]
        
        # Objective: Minimize number of blocks
        model.Minimize(sum(x))
        
        # Constraints: Every t-subset must be covered by at least one block
        # For each t-subset i, sum(x_j for j in ts_to_blocks[i]) >= 1
        for i in range(num_t_subsets):
            blocks_covering_i = ts_to_blocks[i]
            if blocks_covering_i:
                model.Add(sum(x[j] for j in blocks_covering_i) >= 1)
            else:
                # This case shouldn't happen for valid v,k,t, but just in case
                return []
                
        # Set initial upper bound to help the solver
        # The solver will try to find a solution with cost < best_greedy_len
        # Actually, we can just let it run, but setting an initial solution helps.
        # We can also use "AddHint" or just let it search.
        # To be safe, we can set a strict time limit.
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = deadline - time.time() - 5.0
        solver.parameters.num_workers = 8  # Use multiple threads
        
        # Optional: Provide the greedy solution as a hint
        # This can significantly speed up the solver.
        # We need to map the greedy solution blocks back to indices.
        # This is O(N^2) if we do it naively, but N is small enough.
        # Let's build a map of tuple(block) -> index for the greedy solution.
        greedy_block_set = set(tuple(b) for b in best_greedy_sol)
        # We can add hints for the first few variables, but it's complex.
        # Instead, we just run the solver.
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract solution
            sol = []
            for j in range(num_k_blocks):
                if solver.Value(x[j]) == 1:
                    sol.append(list(all_k_blocks[j]))
            
            # Verify validity (just in case)
            if len(sol) < best_greedy_len:
                # Quick check: is it a valid cover?
                # The solver guarantees it, but let's be paranoid.
                # However, checking is expensive. We trust the solver.
                return sol
            else:
                # Solver didn't beat greedy, return best greedy
                return best_greedy_sol
        else:
            # Solver didn't find a solution or timed out without improving
            return best_greedy_sol
    else:
        # Fallback: Enhanced Local Search if OR-Tools is missing
        # This is a structural change from the original code's local search.
        # We use a "Simulated Annealing" style approach with restarts.
        
        def local_search(blocks):
            # Precompute cover counts
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            block_indices_in_sol = {}
            for i, b in enumerate(blocks):
                block_indices_in_sol[tuple(b)] = i
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
            
            improved = True
            while improved:
                improved = False
                # Try to remove redundant blocks
                for i in range(len(blocks)):
                    if time.time() > deadline - 1.0:
                        return blocks
                    redundant = True
                    for ts in itertools.combinations(blocks[i], t):
                        if cover_count[ts_index_map[ts]] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        for ts in itertools.combinations(blocks[i], t):
                            cover_count[ts_index_map[ts]] -= 1
                        improved = True
                        break
                
                # Try swaps (1-exchange)
                if not improved:
                    for i in range(len(blocks)):
                        if time.time() > deadline - 1.0:
                            return blocks
                        unique_ts = []
                        for ts in itertools.combinations(blocks[i], t):
                            idx = ts_index_map[ts]
                            if cover_count[idx] == 1:
                                unique_ts.append(idx)
                        
                        if not unique_ts:
                            continue
                        
                        # Find candidates
                        candidates = set(ts_to_blocks[unique_ts[0]])
                        for ts_idx in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks[ts_idx])
                            if not candidates:
                                break
                        
                        if not candidates:
                            continue
                        
                        valid_candidates = []
                        for c_idx in candidates:
                            if tuple(all_k_blocks[c_idx]) not in block_indices_in_sol:
                                valid_candidates.append(c_idx)
                        
                        if not valid_candidates:
                            continue
                        
                        new_block_idx = random.choice(valid_candidates)
                        new_block = list(all_k_blocks[new_block_idx])
                        
                        # Perform swap
                        old_block = blocks[i]
                        blocks[i] = new_block
                        
                        # Update cover counts
                        for ts in itertools.combinations(old_block, t):
                            cover_count[ts_index_map[ts]] -= 1
                        for ts in itertools.combinations(new_block, t):
                            cover_count[ts_index_map[ts]] += 1
                        
                        improved = True
                        break
            
            return blocks

        # Run local search with multiple restarts
        best_sol = best_greedy_sol
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            # Perturb the solution
            sol = list(best_sol)
            # Randomly replace 5-10 blocks
            num_replacements = random.randint(5, 10)
            for _ in range(num_replacements):
                idx = random.randint(0, len(sol) - 1)
                # Pick a random new block
                new_block = list(random.choice(all_k_blocks))
                sol[idx] = new_block
            
            # Optimize
            sol = local_search(sol)
            if len(sol) < len(best_sol):
                best_sol = sol
                
        return best_sol

