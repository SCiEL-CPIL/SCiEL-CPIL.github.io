import itertools
import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get initial duals (weights)
    duals = None
    try:
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 30})

        if res.success:
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # 3. Primal-Dual Greedy with multiple restarts
    # The primal-dual algorithm: raise duals on uncovered sets until some block's
    # cost is "paid" by the duals of the sets it covers, then select that block.
    # We use the LP duals as initial weights to guide the process.
    
    num_restarts = 300
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with LP duals + adaptive noise
        if restart == 0:
            gains = duals.copy()
        else:
            noise_frac = max(0.02, 0.10 * (1 - restart / num_restarts))
            noise_scale = np.max(duals) * noise_frac
            gains = duals + rng.uniform(0, noise_scale, n_t)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        while True:
            if time.time() > deadline - 2:
                break

            if not np.any(uncovered_mask):
                break

            # Compute marginal gain for each block: sum of duals of uncovered subsets it covers
            # Vectorized: for each block, sum gains over its subsets where uncovered
            # We use the sparse structure: M @ gains gives sum of gains for each block
            # But we only want uncovered ones, so: M @ (gains * uncovered_mask)
            marginal = M @ (gains * uncovered_mask)
            
            best_idx = np.argmax(marginal)
            
            if marginal[best_idx] <= 1e-12:
                # No block can improve; break
                break

            solution_blocks.append(best_idx)
            
            subs_b = subsets_of_block[best_idx]
            newly_covered = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered:
                continue

            for s_idx in newly_covered:
                uncovered_mask[s_idx] = False
                cover_count[s_idx] += 1

            # Primal-dual step: raise duals for uncovered sets
            # The block is "selected" when its cost (1.0) is balanced by accumulated duals
            # We reduce the effective gain for subsets already covered by this block
            # to prevent double-counting in future selections
            for s_idx in subs_b:
                if uncovered_mask[s_idx]:
                    gains[s_idx] *= 0.5  # Dampen to avoid re-selecting similar blocks

            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 4. Fallback if no solution found
    if best_solution is None:
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(best_solution)

    # 5. Post-processing: Deletion Pass + Local Search
    if best_solution is not None and time.time() < deadline - 1:
        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        # Deletion pass
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            best_solution = new_solution
            best_size = len(best_solution)

        # Local search: try to replace one block with another
        max_local_rounds = 5
        for _ in range(max_local_rounds):
            if time.time() > deadline - 1:
                break
            
            improved = False
            sol_set = set(best_solution)
            
            for pos, b_idx in enumerate(list(best_solution)):
                if time.time() > deadline - 1:
                    break
                
                unique_subs = [s for s in subsets_of_block[b_idx] if cover_count[s] == 1]
                
                if not unique_subs:
                    cover_count[subsets_of_block[b_idx]] -= 1
                    best_solution.pop(pos)
                    best_size -= 1
                    improved = True
                    break
                
                for cand in range(n_k):
                    if cand in sol_set:
                        continue
                    
                    cand_subs_set = set(subsets_of_block[cand])
                    if all(s in cand_subs_set for s in unique_subs):
                        temp_cover = cover_count.copy()
                        temp_cover[subsets_of_block[b_idx]] -= 1
                        temp_cover[subsets_of_block[cand]] += 1
                        
                        if np.all(temp_cover >= 1):
                            cover_count = temp_cover
                            best_solution[pos] = cand
                            sol_set.discard(b_idx)
                            sol_set.add(cand)
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break

    return [list(k_subsets[idx]) for idx in best_solution]
