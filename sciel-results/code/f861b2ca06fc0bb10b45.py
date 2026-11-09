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
    
    # Convert to numpy arrays for fast operations
    block_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_tsubs]
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    def random_valid_covering(target_size, deadline):
        """
        Generate a random covering of size approx target_size and repair it to be valid.
        """
        selected = random.sample(range(N_blocks), target_size)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
        
        # Repair: add blocks to cover uncovered t-subsets
        uncovered_mask = (cov == 0).nonzero()[0]
        while len(uncovered_mask) > 0 and time.time() < deadline:
            # Pick a random uncovered t-subset
            target_si = int(uncovered_mask[random.randint(0, len(uncovered_mask)-1)])
            candidate_blocks = tsub_blocks[target_si]
            
            # Find the block that covers the most currently uncovered t-subsets
            best_b = -1
            max_cov = 0
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                count = int(np.sum(cov[block_tsubs_np[b_idx]] == 0))
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                # Fallback: pick any block covering target_si
                best_b = random.choice(candidate_blocks)
                if best_b in selected_set:
                    continue
            
            selected.append(best_b)
            selected_set.add(best_b)
            cov[block_tsubs_np[best_b]] += 1
            uncovered_mask = (cov == 0).nonzero()[0]
        
        return selected, cov

    def local_search_reduce(selected, cov, time_limit=5.0):
        """
        Intensive local search to reduce the number of blocks.
        """
        if not selected:
            return selected, cov
            
        start = time.time()
        selected_set = set(selected)
        
        max_iterations = 20000
        no_improve_streak = 0
        
        for iteration in range(max_iterations):
            if time.time() - start > time_limit or time.time() > DEADLINE:
                break
            
            improved = False
            
            # Strategy 1: Remove redundant blocks
            # Check a random subset of blocks to see if any is redundant
            check_size = min(20, len(selected))
            for _ in range(check_size):
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
            
            # Strategy 2: 1-for-1 swap
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
                            if covered >= would_uncover_count * 0.8:  # Only consider good candidates
                                extra = int(np.sum(cov[subs_add] == 0))
                                score = covered + extra * 0.1
                                scored.append((score, b_add))
                        
                        if scored:
                            scored.sort(reverse=True)
                            # Try top 3
                            for score, best_b_add in scored[:3]:
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
            
            if not improved:
                no_improve_streak += 1
                if no_improve_streak > 100:
                    break
        
        return selected, cov

    # Main loop
    best_sol_idx = None
    best_len = float('inf')
    
    # Try different target sizes for random initialization
    # Poisson optimal size is around 264-270
    target_sizes = [264, 270, 280, 300]
    
    for target_size in target_sizes:
        if time.time() > DEADLINE - 10:
            break
            
        # Multiple restarts for each target size
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            
            sol_idx, cov = random_valid_covering(target_size, DEADLINE - 10)
            
            if len(sol_idx) < best_len:
                best_sol_idx = sol_idx
                best_len = len(sol_idx)
            
            # Local search to reduce
            remaining = DEADLINE - time.time() - 5
            if remaining > 2:
                polished_idx, polished_cov = local_search_reduce(sol_idx, cov, time_limit=remaining)
                if len(polished_idx) < best_len:
                    best_sol_idx = polished_idx
                    best_len = len(polished_idx)
    
    # Final polish on the best solution found
    if best_sol_idx is not None and time.time() < DEADLINE - 2:
        # Reconstruct cov for best_sol_idx
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in best_sol_idx:
            cov[block_tsubs_np[b_idx]] += 1
            
        remaining = DEADLINE - time.time() - 1
        if remaining > 1:
            final_idx, _ = local_search_reduce(best_sol_idx, cov, time_limit=remaining)
            if len(final_idx) < best_len:
                best_sol_idx = final_idx
                best_len = len(final_idx)

    return [all_blocks[i] for i in best_sol_idx]
