"""Covering design C(v,k,t) — Probabilistic construction with repair and local search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 110.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert tsub_blocks to numpy arrays for faster vectorized updates
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]
    
    # Expected number of uncovered t-subsets when choosing m random blocks
    # P(a given t-subset is NOT covered by one block) = 1 - C(k-t, v-t) / C(k, v)
    # Actually: P(covered by one block) = C(k-t, v-t) / C(v, t) ... no
    # P(a specific t-subset S is contained in a random k-subset B) = C(v-t, k-t) / C(v, k)
    # So P(not covered by one block) = 1 - C(v-t, k-t)/C(v,k)
    # P(not covered by m blocks) = (1 - C(v-t,k-t)/C(v,k))^m
    # Expected uncovered = N_t * (1 - C(v-t,k-t)/C(v,k))^m
    
    from math import comb, log, exp
    p_cover = comb(v - t, k - t) / comb(v, k)
    log_uncovered_factor = log(1 - p_cover) if p_cover < 1 else -100
    
    def expected_uncovered(m):
        return N_t * exp(m * log_uncovered_factor)
    
    # Find m such that expected uncovered ~ 1
    # m = log(1/N_t) / log(1 - p_cover)
    if p_cover > 0 and p_cover < 1:
        m_target = int(-log(N_t) / log_uncovered_factor) + 5
    else:
        m_target = N_t  # fallback
    
    m_target = max(m_target, 50)  # minimum reasonable size
    m_target = min(m_target, N_blocks)  # can't exceed total blocks
    
    def repair_greedy(selected_set, uncovered_indices):
        """Add blocks to cover all uncovered t-subsets."""
        selected_list = list(selected_set)
        uncovered = set(uncovered_indices)
        
        if not uncovered:
            return selected_list
        
        # Compute current coverage
        cov = np.zeros(N_t, dtype=np.int32)
        for bi in selected_list:
            for si in block_tsubs[bi]:
                cov[si] += 1
        
        # Greedily add blocks to cover remaining uncovered t-subsets
        while uncovered and time.time() < DEADLINE:
            # Find block that covers most uncovered t-subsets
            best_block = -1
            best_count = 0
            for bi in range(N_blocks):
                if bi in selected_set:
                    continue
                count = 0
                for si in block_tsubs[bi]:
                    if si in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = bi
            
            if best_block == -1 or best_count == 0:
                break
            
            selected_list.append(best_block)
            selected_set.add(best_block)
            for si in block_tsubs[best_block]:
                if si in uncovered:
                    uncovered.discard(si)
                cov[si] += 1
        
        return selected_list
    
    def try_reduce(blocks):
        if not blocks:
            return blocks
        
        # Precompute subs for each block in the solution
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = np.zeros(N_t, dtype=np.int32)
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs[i]:
                        cov[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]
    
    def local_search(blocks, max_iter=1000):
        """Try to improve by removing redundant ones and swapping."""
        if time.time() > DEADLINE:
            return blocks
        
        n = len(blocks)
        
        # Compute coverage
        cov = np.zeros(N_t, dtype=np.int32)
        block_subs_list = []
        for b in blocks:
            subs = [t_idx[s] for s in itertools.combinations(b, t)]
            block_subs_list.append(subs)
            for si in subs:
                cov[si] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < DEADLINE:
            improved = False
            iteration += 1
            
            # Try removing each block
            for i in range(n):
                if time.time() > DEADLINE:
                    return blocks
                subs = block_subs_list[i]
                can_remove = all(cov[si] > 1 for si in subs)
                if can_remove:
                    for si in subs:
                        cov[si] -= 1
                    blocks = blocks[:i] + blocks[i+1:]
                    block_subs_list = block_subs_list[:i] + block_subs_list[i+1:]
                    n = len(blocks)
                    improved = True
                    break
            
            # Try swapping: remove block i, add block j that covers more
            if not improved and time.time() < DEADLINE:
                for i in range(n):
                    if time.time() > DEADLINE:
                        return blocks
                    subs_i = set(block_subs_list[i])
                    # t-subsets that would become uncovered if we remove block i
                    would_uncover = [si for si in subs_i if cov[si] == 1]
                    if not would_uncover:
                        continue
                    
                    # Find a block j (not in current set) that covers all would_uncover
                    current_set = set(blocks)
                    for j in range(N_blocks):
                        if j in current_set:
                            continue
                        subs_j = set(block_tsubs[j])
                        if all(si in subs_j for si in would_uncover):
                            # Check if this swap actually helps (j covers more overall)
                            new_cov_gain = 0
                            for si in subs_j:
                                if si in subs_i:
                                    continue
                                if cov[si] == 0:
                                    new_cov_gain += 1
                            
                            # Accept if j covers at least as many unique t-subsets
                            if new_cov_gain >= 0:
                                # Perform swap
                                for si in subs_i:
                                    cov[si] -= 1
                                for si in subs_j:
                                    cov[si] += 1
                                blocks[i] = j
                                block_subs_list[i] = list(subs_j)
                                improved = True
                                break
                    if improved:
                        break
        
        return blocks
    
    best_sol = None
    best_len = float('inf')
    
    # Phase 1: Probabilistic construction with repair
    # Try several different m values around the target
    m_values = sorted(set([
        m_target,
        max(50, m_target - 10),
        m_target + 10,
        m_target + 20,
        max(50, m_target - 20),
    ]))
    
    for m in m_values:
        if time.time() > DEADLINE - 5.0:
            break
        
        m = max(50, min(m, N_blocks))
        
        # Randomly select m blocks
        selected_indices = random.sample(range(N_blocks), m)
        selected_set = set(selected_indices)
        
        # Find uncovered t-subsets
        cov = np.zeros(N_t, dtype=np.int8)
        for bi in selected_indices:
            for si in block_tsubs[bi]:
                cov[si] = 1
        
        uncovered = [i for i in range(N_t) if cov[i] == 0]
        
        # Repair
        repaired = repair_greedy(selected_set, uncovered)
        
        # Convert to actual blocks
        sol = [all_blocks[bi] for bi in repaired]
        
        # Reduce and local search
        sol = try_reduce(sol)
        sol = local_search(sol, max_iter=300)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
    
    # Phase 2: Greedy set cover as baseline
    if time.time() < DEADLINE - 10.0:
        def run_greedy(randomize=False, seed=None):
            if seed is not None:
                random.seed(seed)
            
            # Initial gains: number of t-subsets covered by each block
            block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
            is_covered = np.zeros(N_t, dtype=np.int8)
            uncovered_count = N_t
            selected_blocks = []
            
            while uncovered_count > 0 and time.time() < DEADLINE:
                if randomize:
                    max_gain = np.max(block_gain)
                    if max_gain <= 0:
                        break
                    # Find indices with max gain
                    candidates = np.where(block_gain == max_gain)[0]
                    best_block_idx = random.choice(candidates.tolist())
                else:
                    max_gain = np.max(block_gain)
                    if max_gain <= 0:
                        break
                    best_block_idx = np.argmax(block_gain)
                
                selected_blocks.append(all_blocks[best_block_idx])
                
                # Update coverage and gains
                subs = block_tsubs[best_block_idx]
                for si in subs:
                    if is_covered[si] == 0:
                        is_covered[si] = 1
                        uncovered_count -= 1
                        # Vectorized update of gains for blocks covering this t-subset
                        block_gain[tsub_blocks_np[si]] -= 1
                        
            return selected_blocks
        
        sol = run_greedy(randomize=False)
        sol = try_reduce(sol)
        sol = local_search(sol, max_iter=500)
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
    
    # Phase 3: Randomized restarts with probabilistic construction
    restart_count = 0
    while time.time() < DEADLINE - 3.0 and best_sol is not None:
        seed = random.randint(0, 2**31)
        rng = random.Random(seed)
        
        # Vary m slightly
        m = max(50, min(m_target + rng.randint(-15, 25), N_blocks))
        
        selected_indices = rng.sample(range(N_blocks), m)
        selected_set = set(selected_indices)
        
        cov = np.zeros(N_t, dtype=np.int8)
        for bi in selected_indices:
            for si in block_tsubs[bi]:
                cov[si] = 1
        
        uncovered = [i for i in range(N_t) if cov[i] == 0]
        
        # Repair
        repaired = repair_greedy(selected_set, uncovered)
        
        sol = [all_blocks[bi] for bi in repaired]
        sol = try_reduce(sol)
        sol = local_search(sol, max_iter=150)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
        restart_count += 1
    
    if best_sol is None:
        # Fallback: pure greedy
        sol = run_greedy(randomize=False)
        sol = try_reduce(sol)
        best_sol = sol
    
    return best_sol
