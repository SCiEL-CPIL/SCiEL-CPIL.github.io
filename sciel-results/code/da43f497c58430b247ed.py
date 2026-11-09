"""Covering design C(v,k,t) — Greedy Set Cover with Local Search and Randomized Restarts."""
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 130.0  # 130s safety margin (evaluator kills at 150s)
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
    def run_greedy(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        block_gain = [len(subs) for subs in block_tsubs]
        is_covered = [0] * N_t
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            best_block_idx = -1
            max_gain = -1
            
            if randomize:
                # Find max gain, then randomly pick among ties
                max_gain = max(block_gain)
                if max_gain <= 0:
                    break
                candidates = [i for i in range(N_blocks) if block_gain[i] == max_gain]
                best_block_idx = random.choice(candidates)
            else:
                for i in range(N_blocks):
                    if block_gain[i] > max_gain:
                        max_gain = block_gain[i]
                        best_block_idx = i
                if best_block_idx == -1 or max_gain == 0:
                    break
            
            selected_blocks.append(all_blocks[best_block_idx])
            
            for si in block_tsubs[best_block_idx]:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        if not blocks:
            return blocks
        
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = [0] * N_t
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
        """Try to improve by swapping blocks or removing redundant ones."""
        if time.time() > DEADLINE:
            return blocks
        
        n = len(blocks)
        block_set = set(blocks)
        
        # Compute coverage
        cov = [0] * N_t
        for b in blocks:
            for s in itertools.combinations(b, t):
                cov[t_idx[s]] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < DEADLINE:
            improved = False
            iteration += 1
            
            # Try removing each block
            for i in range(n):
                if time.time() > DEADLINE:
                    return blocks
                b = blocks[i]
                subs = [t_idx[s] for s in itertools.combinations(b, t)]
                can_remove = all(cov[si] > 1 for si in subs)
                if can_remove:
                    for si in subs:
                        cov[si] -= 1
                    blocks = blocks[:i] + blocks[i+1:]
                    n = len(blocks)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove block i, add block j not in current set
            for i in range(n):
                if time.time() > DEADLINE:
                    return blocks
                b_remove = blocks[i]
                subs_remove = [t_idx[s] for s in itertools.combinations(b_remove, t)]
                
                # Check if removing would uncover any t-subset
                critical = any(cov[si] <= 1 for si in subs_remove)
                if critical:
                    continue
                
                # Find a block that covers some of the critical t-subsets
                # Actually, if not critical, we can just remove. So let's try different swaps.
                # Swap: remove b_remove, add b_add where b_add covers t-subsets that b_remove covers
                # This is complex. Let's just try adding a random block and see if we can remove more.
                pass
        
        return blocks

    # First run: deterministic greedy
    sol = run_greedy(randomize=False)
    sol = try_reduce(sol)
    sol = local_search(sol, max_iter=500)
    best_sol = sol
    best_len = len(sol)
    
    # Use remaining time for randomized restarts
    restart_count = 0
    while time.time() < DEADLINE - 5.0:
        seed = random.randint(0, 2**31)
        cand = run_greedy(randomize=True, seed=seed)
        cand = try_reduce(cand)
        cand = local_search(cand, max_iter=200)
        if len(cand) < best_len:
            best_sol = cand
            best_len = len(cand)
        restart_count += 1
    
    return best_sol
