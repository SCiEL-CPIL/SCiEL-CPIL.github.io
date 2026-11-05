import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0
    
    # 1. Precomputation
    # All t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # All k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Helper: Compute coverage counts
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
        return cov

    # Helper: Reduce solution (remove redundant blocks)
    def reduce_solution(sel):
        if not sel: return []
        cov = compute_cov(sel)
        
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    def greedy_cover(randomize=False, conflict_aware=False):
        """Greedy set cover with optional randomization and conflict-aware scoring."""
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])
            
        selected = []
        remaining = n_tsub
        selected_set = set()
        
        # For conflict-aware: track how many t-sets are already covered by selected blocks
        # We want to maximize new_coverage / (1 + overlap_penalty)
        
        while remaining > 0:
            if time.time() > deadline - 20.0:
                break
            
            if conflict_aware:
                # Conflict-aware: score = new_gain / (1 + overlap_with_selected)
                # overlap_with_selected = number of t-sets in block that are already covered
                best_score = -1
                best_bi = -1
                best_new_gain = 0
                
                # Sample a subset of candidates for speed, or evaluate all if fast enough
                # For v=19, k=8, t=5: n_blocks = C(19,8) = 75582, n_tsub = C(19,5) = 11628
                # Evaluating all blocks each step might be too slow. Let's try a hybrid.
                
                if len(selected) < 50:
                    # Early: evaluate all blocks
                    for bi in range(n_blocks):
                        if gain[bi] == 0:
                            continue
                        # Count overlap: t-sets in block bi that are already covered
                        overlap = 0
                        for ti in block_tsets[bi]:
                            if not uncovered[ti]:
                                overlap += 1
                        score = gain[bi] / (1.0 + overlap * 0.1)
                        if score > best_score:
                            best_score = score
                            best_bi = bi
                            best_new_gain = gain[bi]
                else:
                    # Later: sample candidates from those with high gain
                    mask = gain > 0
                    if not np.any(mask):
                        break
                    # Get top candidates by gain
                    gains_arr = gain[mask]
                    # Sample up to 100 candidates
                    if len(gains_arr) > 100:
                        # Take top 100 by gain
                        top_indices = np.argsort(gains_arr)[-100:]
                        candidate_indices = np.where(mask)[0][top_indices]
                    else:
                        candidate_indices = np.where(mask)[0]
                    
                    for bi in candidate_indices:
                        overlap = 0
                        for ti in block_tsets[bi]:
                            if not uncovered[ti]:
                                overlap += 1
                        score = gain[bi] / (1.0 + overlap * 0.1)
                        if score > best_score:
                            best_score = score
                            best_bi = bi
                            best_new_gain = gain[bi]
                
                if best_bi == -1 or best_new_gain == 0:
                    break
                bi = best_bi
                
            elif randomize:
                mask = gain > 0
                if not np.any(mask):
                    break
                max_gain = gain[mask].max()
                candidates = np.where(mask & (gain == max_gain))[0]
                bi = int(random.choice(candidates))
            else:
                bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            selected.append(bi)
            selected_set.add(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0:
                            gain[bj] -= 1
        return selected

    def local_search(sel, max_iter=200):
        """Local search: 1-for-1 swaps and 1-for-2 swaps to reduce block count."""
        sel = list(sel)
        sel_set = set(sel)
        cov = compute_cov(sel)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # First, remove any redundant blocks
            new_sel = []
            for bi in sel:
                essential = False
                for ti in block_tsets[bi]:
                    if cov[ti] == 1:
                        essential = True
                        break
                if not essential:
                    for ti in block_tsets[bi]:
                        cov[ti] -= 1
                else:
                    new_sel.append(bi)
            
            if len(new_sel) < len(sel):
                sel = new_sel
                sel_set = set(sel)
                cov = compute_cov(sel)
                improved = True
                continue
            
            # Compute unique t-sets for each selected block
            unique_sets = {}
            for bi in sel:
                u = frozenset(ti for ti in block_tsets[bi] if cov[ti] == 1)
                unique_sets[bi] = u
            
            # Try 1-for-2 swap
            sorted_blocks = sorted(sel, key=lambda bi: len(unique_sets[bi]))
            found_move = False
            for i in range(len(sorted_blocks)):
                if time.time() > deadline - 5.0:
                    break
                bi = sorted_blocks[i]
                u_i = unique_sets[bi]
                if len(u_i) > 4:
                    continue
                
                for j in range(i + 1, len(sorted_blocks)):
                    bj = sorted_blocks[j]
                    u_j = unique_sets[bj]
                    if len(u_j) > 4:
                        continue
                    
                    target = u_i | u_j
                    if not target:
                        continue
                    
                    first_t = next(iter(target))
                    candidates = set(tset_to_blocks[first_t])
                    for t in list(target)[1:]:
                        candidates.intersection_update(tset_to_blocks[t])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    candidates.difference_update(sel_set)
                    if not candidates:
                        continue
                    
                    b_new = next(iter(candidates))
                    
                    new_block_tsets = set(block_tsets[b_new])
                    if not (u_i <= new_block_tsets and u_j <= new_block_tsets):
                        continue
                    
                    redundant_count = 2
                    for bk in sel:
                        if bk == bi or bk == bj:
                            continue
                        u_k = unique_sets[bk]
                        if u_k and u_k <= new_block_tsets:
                            redundant_count += 1
                    
                    if redundant_count > 1:
                        sel.remove(bi)
                        sel.remove(bj)
                        for bk in list(sel):
                            if bk != b_new and unique_sets[bk] and unique_sets[bk] <= new_block_tsets:
                                sel.remove(bk)
                        sel.append(b_new)
                        sel_set = set(sel)
                        cov = compute_cov(sel)
                        improved = True
                        found_move = True
                        break
                if found_move:
                    break
            
            if not found_move:
                # Try 1-for-1 swap
                for bi in sel:
                    if time.time() > deadline - 5.0:
                        break
                    u_i = unique_sets[bi]
                    if not u_i:
                        continue
                    if len(u_i) > 5:
                        continue
                    
                    first_t = next(iter(u_i))
                    candidates = set(tset_to_blocks[first_t])
                    for t in list(u_i)[1:]:
                        candidates.intersection_update(tset_to_blocks[t])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    candidates.discard(bi)
                    if not candidates:
                        continue
                    
                    best_b_new = None
                    best_extra = 0
                    for b_new in list(candidates)[:20]:
                        extra = 0
                        for bk in sel:
                            if bk == bi:
                                continue
                            u_k = unique_sets[bk]
                            if u_k and u_k <= set(block_tsets[b_new]):
                                extra += 1
                        if extra > best_extra:
                            best_extra = extra
                            best_b_new = b_new
                    
                    if best_b_new is not None and best_extra > 0:
                        redundant = [bi]
                        for bk in sel:
                            if bk == bi:
                                continue
                            u_k = unique_sets[bk]
                            if u_k and u_k <= set(block_tsets[best_b_new]):
                                redundant.append(bk)
                        
                        for bk in redundant:
                            sel.remove(bk)
                        sel.append(best_b_new)
                        sel_set = set(sel)
                        cov = compute_cov(sel)
                        improved = True
                        break
            
            if not improved:
                break
        
        return sel

    def simulated_annealing(initial_blocks, time_limit):
        """
        Simulated Annealing for covering design optimization.
        State: list of block indices
        Energy: len(blocks) + penalty * num_uncovered
        Moves: remove, swap, add
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        # Initial coverage
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in block_tsets[b_idx]:
                cover_count[t_idx] += 1
        
        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 100.0  # High penalty to ensure validity
        current_energy = len(current_blocks) + penalty * num_uncovered
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        rng = random.Random(42)
        T = 50.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Precompute unique sets for current state to speed up swap checks
        # We'll update this dynamically
        
        while time.time() < time_limit and T > min_T:
            move_type = rng.random()
            
            if move_type < 0.30 and len(current_blocks) > 0:
                # Try removal
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                new_uncovered = 0
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        new_uncovered += 1
                
                if new_uncovered == 0:
                    # Valid removal
                    for t_idx in block_tsets[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    # Invalid removal - accept with probability
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        for t_idx in block_tsets[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            elif move_type < 0.90 and len(current_blocks) > 0:
                # Try swap
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                # Find unique t-subsets (covered only by old_b)
                unique_ts = []
                for t_idx in block_tsets[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    # Candidate must cover all unique t-subsets
                    candidates = set(tset_to_blocks[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(tset_to_blocks[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    # No unique subsets, try random candidates that cover many uncovered
                    # Or just random
                    candidates = set()
                    for _ in range(5):
                        c = rng.randint(0, n_blocks - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                
                if candidates:
                    new_b = rng.choice(list(candidates))
                    
                    # Calculate energy change
                    # Remove old_b
                    old_uncovered_delta = 0
                    for t_idx in block_tsets[old_b]:
                        if cover_count[t_idx] == 1:
                            old_uncovered_delta += 1
                    
                    # Add new_b
                    new_uncovered_delta = 0
                    for t_idx in block_tsets[new_b]:
                        if cover_count[t_idx] == 0:
                            new_uncovered_delta += 1
                        elif cover_count[t_idx] == 1:
                            # If it was unique to old_b, it's still covered by new_b
                            # But we need to check if it was covered by old_b AND new_b
                            pass 
                    
                    # This simple delta calculation is tricky. Let's do a proper check.
                    # Simulate the move
                    temp_cover = cover_count.copy()
                    for t_idx in block_tsets[old_b]:
                        temp_cover[t_idx] -= 1
                    for t_idx in block_tsets[new_b]:
                        temp_cover[t_idx] += 1
                    
                    temp_uncovered = int(np.sum(temp_cover == 0))
                    temp_energy = len(current_blocks) + penalty * temp_uncovered
                    delta_E = temp_energy - current_energy
                    
                    if rng.random() < np.exp(-delta_E / T):
                        # Apply move
                        cover_count = temp_cover
                        current_blocks[i] = new_b
                        current_energy = temp_energy

            else:
                # Try add (if we have uncovered sets)
                if num_uncovered > 0:
                    # Find block covering most uncovered
                    best_gain = -1
                    best_b = -1
                    for _ in range(10): # Sample 10 random blocks
                        b_idx = rng.randint(0, n_blocks - 1)
                        if b_idx in current_blocks:
                            continue
                        gain = 0
                        for t_idx in block_tsets[b_idx]:
                            if cover_count[t_idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b_idx
                    
                    if best_b != -1 and best_gain > 0:
                        # Apply add
                        current_blocks.append(best_b)
                        for t_idx in block_tsets[best_b]:
                            cover_count[t_idx] += 1
                        num_uncovered -= best_gain
                        current_energy += 1.0 - penalty * best_gain

            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
                
            T *= cooling_rate

        return best_blocks

    # Main execution
    best_sol = None
    best_size = float('inf')
    
    # 1. Conflict-Aware Greedy
    sol = greedy_cover(randomize=False, conflict_aware=True)
    sol = reduce_solution(sol)
    sol = local_search(sol)
    sol = reduce_solution(sol)
    
    if len(sol) < best_size:
        best_size = len(sol)
        best_sol = list(sol)
    
    # 2. Simulated Annealing on the greedy result
    if time.time() < deadline - 10.0:
        sa_time_limit = time.time() + 90.0 # Use 90s for SA
        sol_sa = simulated_annealing(best_sol, sa_time_limit)
        sol_sa = reduce_solution(sol_sa)
        sol_sa = local_search(sol_sa, max_iter=50)
        sol_sa = reduce_solution(sol_sa)
        
        if len(sol_sa) < best_size:
            best_size = len(sol_sa)
            best_sol = list(sol_sa)
    
    # 3. Randomized Conflict-Aware Greedy + SA
    if time.time() < deadline - 10.0:
        random.seed(123)
        sol_r = greedy_cover(randomize=False, conflict_aware=True)
        sol_r = reduce_solution(sol_r)
        sol_r = local_search(sol_r)
        sol_r = reduce_solution(sol_r)
        
        if len(sol_r) < best_size:
            best_size = len(sol_r)
            best_sol = list(sol_r)
            
        # SA on conflict-aware greedy
        if time.time() < deadline - 5.0:
            sa_time_limit = time.time() + 20.0
            sol_sa_r = simulated_annealing(best_sol, sa_time_limit)
            sol_sa_r = reduce_solution(sol_sa_r)
            sol_sa_r = local_search(sol_sa_r, max_iter=20)
            sol_sa_r = reduce_solution(sol_sa_r)
            
            if len(sol_sa_r) < best_size:
                best_size = len(sol_sa_r)
                best_sol = list(sol_sa_r)
    
    # Final polish
    if best_sol is not None and time.time() < deadline - 2.0:
        best_sol = local_search(best_sol, max_iter=100)
        best_sol = reduce_solution(best_sol)
    
    if best_sol is None:
        best_sol = list(range(n_blocks))
    
    return [list(all_blocks[bi]) for bi in best_sol]
