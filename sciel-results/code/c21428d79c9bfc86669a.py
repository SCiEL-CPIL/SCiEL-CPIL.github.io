import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute: for each block, which t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
    
    # For each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
    # Convert to numpy arrays
    block_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_tsubs]
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]
    
    # Precompute block intersection matrix (number of shared t-subsets)
    # This is O(N_blocks^2 * C(8,5)) which is feasible
    # But for memory, we'll compute lazily or use a different approach
    # For now, precompute a sparse representation
    
    def run_greedy_indices(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            
            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = int(np.argmax(block_gain))
                
            selected_indices.append(best_block_idx)
            
            subs_np = block_tsubs_np[best_block_idx]
            newly_covered_mask = is_covered[subs_np] == 0
            newly_covered = subs_np[newly_covered_mask]
            
            if len(newly_covered) > 0:
                is_covered[newly_covered] = 1
                uncovered_count -= len(newly_covered)
                
                all_block_indices = []
                for si in newly_covered:
                    all_block_indices.extend(tsub_blocks_np[si].tolist())
                if all_block_indices:
                    all_block_indices_np = np.array(all_block_indices, dtype=np.int32)
                    np.subtract.at(block_gain, all_block_indices_np, 1)
                    
        return selected_indices

    def local_search_reduce(selected, time_limit=5.0):
        """
        Intensive local search using multiple strategies.
        Uses coverage counts for fast redundancy checks.
        """
        if not selected:
            return selected
            
        start = time.time()
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
        
        # Precompute: for each block in selected, the set of blocks that share t-subsets with it
        # This helps with faster swap evaluation
        
        max_iterations = 5000
        no_improve_streak = 0
        
        for iteration in range(max_iterations):
            if time.time() - start > time_limit or time.time() > DEADLINE:
                break
            
            improved = False
            
            # Strategy 1: Remove redundant blocks (more aggressive)
            for _ in range(10):
                b_idx = random.choice(selected)
                subs_np = block_tsubs_np[b_idx]
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    no_improve_streak = 0
                    break
            
            if improved:
                continue
            
            # Strategy 2: 1-for-1 swap with better scoring
            if len(selected) >= 1:
                b_remove = random.choice(selected)
                subs_remove = block_tsubs_np[b_remove]
                would_uncover_mask = cov[subs_remove] == 1
                would_uncover_count = int(np.sum(would_uncover_mask))
                
                if would_uncover_count > 0:
                    would_uncover = subs_remove[would_uncover_mask]
                    
                    # Find candidate blocks that cover most of would_uncover
                    candidate_set = set()
                    for si in would_uncover:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b_remove)
                    candidate_set -= selected_set
                    
                    if candidate_set:
                        candidates = list(candidate_set)
                        # Score all candidates, pick top ones
                        scored = []
                        for b_add in candidates:
                            subs_add = block_tsubs_np[b_add]
                            covered = int(np.sum(np.isin(subs_add, would_uncover)))
                            if covered >= would_uncover_count * 0.7:  # Only consider good candidates
                                extra = int(np.sum(cov[subs_add] == 0))
                                score = covered + extra * 0.05
                                scored.append((score, b_add))
                        
                        if scored:
                            scored.sort(reverse=True)
                            # Try top 5
                            for score, best_b_add in scored[:5]:
                                if score >= would_uncover_count * 0.95:
                                    subs_add = block_tsubs_np[best_b_add]
                                    selected.remove(b_remove)
                                    selected_set.discard(b_remove)
                                    cov[subs_remove] -= 1
                                    
                                    selected.append(best_b_add)
                                    selected_set.add(best_b_add)
                                    cov[subs_add] += 1
                                    improved = True
                                    no_improve_streak = 0
                                    break
            
            if improved:
                continue
            
            # Strategy 3: 2-for-1 exchange
            if len(selected) >= 2 and time.time() - start < time_limit * 0.7:
                b1, b2 = random.sample(selected, 2)
                subs1 = block_tsubs_np[b1]
                subs2 = block_tsubs_np[b2]
                
                cov_temp = cov.copy()
                cov_temp[subs1] -= 1
                cov_temp[subs2] -= 1
                would_uncover_mask = cov_temp == 0
                would_uncover_count = int(np.sum(would_uncover_mask))
                
                if would_uncover_count > 0:
                    would_uncover = np.where(would_uncover_mask)[0]
                    
                    candidate_set = set()
                    for si in would_uncover:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b1)
                    candidate_set.discard(b2)
                    candidate_set -= selected_set
                    
                    if candidate_set:
                        candidates = list(candidate_set)
                        best_b_add = -1
                        best_coverage = -1
                        
                        for b_add in candidates:
                            subs_add = block_tsubs_np[b_add]
                            coverage = int(np.sum(np.isin(subs_add, would_uncover)))
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_b_add = b_add
                            if coverage >= would_uncover_count:
                                break
                        
                        if best_b_add != -1 and best_coverage >= would_uncover_count:
                            subs_add = block_tsubs_np[best_b_add]
                            selected.remove(b1)
                            selected_set.discard(b1)
                            selected.remove(b2)
                            selected_set.discard(b2)
                            cov[subs1] -= 1
                            cov[subs2] -= 1
                            
                            selected.append(best_b_add)
                            selected_set.add(best_b_add)
                            cov[subs_add] += 1
                            improved = True
                            no_improve_streak = 0
            
            # Strategy 4: 2-for-2 exchange (new)
            if not improved and len(selected) >= 4 and time.time() - start < time_limit * 0.5:
                b1, b2 = random.sample(selected, 2)
                subs1 = block_tsubs_np[b1]
                subs2 = block_tsubs_np[b2]
                
                cov_temp = cov.copy()
                cov_temp[subs1] -= 1
                cov_temp[subs2] -= 1
                would_uncover_mask = cov_temp == 0
                would_uncover_count = int(np.sum(would_uncover_mask))
                
                if would_uncover_count > 0:
                    would_uncover = np.where(would_uncover_mask)[0]
                    
                    candidate_set = set()
                    for si in would_uncover:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b1)
                    candidate_set.discard(b2)
                    candidate_set -= selected_set
                    
                    if len(candidate_set) >= 2:
                        candidates = list(candidate_set)
                        # Find two blocks that together cover all would_uncover
                        best_pair = None
                        best_total = -1
                        
                        # Sample pairs if too many candidates
                        if len(candidates) > 20:
                            candidates = random.sample(candidates, 20)
                        
                        for i in range(len(candidates)):
                            for j in range(i+1, len(candidates)):
                                b_add1 = candidates[i]
                                b_add2 = candidates[j]
                                subs_add1 = block_tsubs_np[b_add1]
                                subs_add2 = block_tsubs_np[b_add2]
                                # Union of coverage
                                combined = np.union1d(subs_add1, subs_add2)
                                coverage = int(np.sum(np.isin(would_uncover, combined)))
                                if coverage > best_total:
                                    best_total = coverage
                                    best_pair = (b_add1, b_add2)
                                if coverage >= would_uncover_count:
                                    break
                            if best_total >= would_uncover_count:
                                break
                        
                        if best_pair is not None and best_total >= would_uncover_count:
                            b_add1, b_add2 = best_pair
                            subs_add1 = block_tsubs_np[b_add1]
                            subs_add2 = block_tsubs_np[b_add2]
                            selected.remove(b1)
                            selected_set.discard(b1)
                            selected.remove(b2)
                            selected_set.discard(b2)
                            cov[subs1] -= 1
                            cov[subs2] -= 1
                            
                            selected.append(b_add1)
                            selected_set.add(b_add1)
                            selected.append(b_add2)
                            selected_set.add(b_add2)
                            cov[subs_add1] += 1
                            cov[subs_add2] += 1
                            improved = True
                            no_improve_streak = 0
            
            if not improved:
                no_improve_streak += 1
                # Try a few more random removals
                for _ in range(5):
                    if not selected:
                        break
                    b_idx = random.choice(selected)
                    subs_np = block_tsubs_np[b_idx]
                    if int(np.sum(cov[subs_np] == 1)) == 0:
                        selected.remove(b_idx)
                        selected_set.discard(b_idx)
                        cov[subs_np] -= 1
                        improved = True
                        no_improve_streak = 0
                        break
                if not improved and no_improve_streak > 50:
                    break
        
        return selected

    def simulated_anneal(initial_indices, time_limit=8.0):
        """
        Simulated Annealing with efficient move generation.
        """
        if not initial_indices:
            return initial_indices
            
        M = 500
        T = 5.0
        alpha = 0.985
        T_min = 0.01
        
        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
            
        uncovered_count = int(np.sum(cov == 0))
        current_cost = len(selected) + M * uncovered_count
        best_cost = current_cost
        best_selected = list(selected)
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            move_type = random.random()
            
            if move_type < 0.35 and selected:
                # Remove a block
                b_idx = random.choice(selected)
                subs_np = block_tsubs_np[b_idx]
                delta_uncovered = int(np.sum(cov[subs_np] == 1))
                delta_cost = -1 + M * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
                        
            elif move_type < 0.55:
                # Add a block covering uncovered t-subsets
                if uncovered_count > 0:
                    uncovered_mask = (cov == 0).nonzero()[0]
                    if len(uncovered_mask) > 0:
                        target_si = int(uncovered_mask[random.randint(0, len(uncovered_mask)-1)])
                        candidate_blocks = tsub_blocks[target_si]
                        if candidate_blocks:
                            b_idx = random.choice(candidate_blocks)
                            if b_idx not in selected_set:
                                subs_np = block_tsubs_np[b_idx]
                                newly_covered = int(np.sum(cov[subs_np] == 0))
                                delta_cost = 1 - M * newly_covered
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    selected.append(b_idx)
                                    selected_set.add(b_idx)
                                    cov[subs_np] += 1
                                    uncovered_count -= newly_covered
                                    current_cost += delta_cost
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
                else:
                    b_idx = random.randint(0, N_blocks - 1)
                    if b_idx not in selected_set:
                        delta_cost = 1
                        if random.random() < np.exp(-delta_cost / T):
                            selected.append(b_idx)
                            selected_set.add(b_idx)
                            cov[block_tsubs_np[b_idx]] += 1
                            current_cost += delta_cost
                            
            else:
                # Swap: remove one, add another (improved)
                if len(selected) >= 1:
                    b_remove = random.choice(selected)
                    subs_remove = block_tsubs_np[b_remove]
                    would_uncover_mask = cov[subs_remove] == 1
                    would_uncover_count = int(np.sum(would_uncover_mask))
                    
                    if would_uncover_count > 0:
                        would_uncover = subs_remove[would_uncover_mask]
                        
                        # Build candidate set from all would_uncover
                        candidate_set = set()
                        for si in would_uncover:
                            candidate_set.update(tsub_blocks[si])
                        candidate_set.discard(b_remove)
                        candidate_set -= selected_set
                        
                        if candidate_set:
                            candidates = list(candidate_set)
                            sample = candidates if len(candidates) <= 10 else random.sample(candidates, 10)
                            best_b_add = -1
                            best_score = -1
                            
                            for b_add in sample:
                                subs_add = block_tsubs_np[b_add]
                                score = int(np.sum(np.isin(subs_add, would_uncover)))
                                if score > best_score:
                                    best_score = score
                                    best_b_add = b_add
                            
                            if best_b_add != -1 and best_score >= would_uncover_count * 0.8:
                                subs_add = block_tsubs_np[best_b_add]
                                delta_cover_add = int(np.sum((cov[subs_add] == 0) | (np.isin(subs_add, would_uncover) & (cov[subs_add] == 1))))
                                net_uncover_change = would_uncover_count - delta_cover_add
                                delta_cost = net_uncover_change * M
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    selected.remove(b_remove)
                                    selected_set.discard(b_remove)
                                    cov[subs_remove] -= 1
                                    uncovered_count += would_uncover_count
                                    
                                    selected.append(best_b_add)
                                    selected_set.add(best_b_add)
                                    cov[subs_add] += 1
                                    uncovered_count -= delta_cover_add
                                    current_cost += delta_cost
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
            
            T *= alpha
        
        # Repair
        uncovered_mask = (cov == 0).nonzero()[0]
        while len(uncovered_mask) > 0 and time.time() < DEADLINE:
            best_b = -1
            max_cov = 0
            candidate_blocks = set()
            for si in uncovered_mask[:20]:
                candidate_blocks.update(tsub_blocks[si])
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                count = int(np.sum(cov[block_tsubs_np[b_idx]] == 0))
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                break
                
            selected.append(best_b)
            selected_set.add(best_b)
            cov[block_tsubs_np[best_b]] += 1
            uncovered_mask = (cov == 0).nonzero()[0]
        
        # Final reduction
        remaining = min(3.0, DEADLINE - time.time() - 1.5)
        if remaining > 0.5:
            selected = local_search_reduce(selected, time_limit=remaining)
        
        return selected

    # Main loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = simulated_anneal(sol_idx, time_limit=10.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        remaining = DEADLINE - time.time()
        if remaining < 12:
            break
        time_limit = min(6.0, remaining - 12.0)
        if time_limit < 1.0:
            break
            
        cand_idx = simulated_anneal(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1
        
        # If we've found a good solution, do a quick local search polish
        if restart_count > 0 and restart_count % 3 == 0 and time.time() < DEADLINE - 10:
            selected = list(best_sol_idx)
            polish_time = min(2.0, DEADLINE - time.time() - 8.0)
            if polish_time > 0.5:
                polished = local_search_reduce(selected, time_limit=polish_time)
                if len(polished) < best_len:
                    best_sol_idx = polished
                    best_len = len(polished)

    # 3. Final polish
    if best_sol_idx is not None and time.time() < DEADLINE - 2:
        selected = list(best_sol_idx)
        polish_time = min(8.0, DEADLINE - time.time() - 1.5)
        if polish_time > 1.0:
            # Multiple rounds of local search with decreasing time
            round1_time = polish_time * 0.3
            selected = local_search_reduce(selected, time_limit=round1_time)
            
            if time.time() < DEADLINE - 2:
                remaining = DEADLINE - time.time() - 2.0
                if remaining > 2.0:
                    polished = simulated_anneal(selected, time_limit=min(4.0, remaining))
                    if len(polished) < len(selected):
                        selected = polished
            
            if time.time() < DEADLINE - 1:
                remaining = DEADLINE - time.time() - 1.0
                if remaining > 0.5:
                    selected = local_search_reduce(selected, time_limit=remaining)
            
            # One more final reduction pass
            if time.time() < DEADLINE - 0.5:
                remaining = DEADLINE - time.time() - 0.5
                if remaining > 0.3:
                    selected = local_search_reduce(selected, time_limit=remaining)
            
            if len(selected) < best_len:
                best_sol_idx = selected
                best_len = len(selected)

    return [all_blocks[i] for i in best_sol_idx]
