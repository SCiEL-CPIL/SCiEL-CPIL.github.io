import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    # Precompute
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
    # block_cover[bi] = list of t-subset indices covered by block bi
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: Exact Minimum Set Cover using Branch and Bound
    # candidates: list of block indices
    # target_set: set of t-subset indices to cover
    # Returns: list of block indices forming a minimal cover
    def min_set_cover(candidates, target_set):
        if not target_set:
            return []
        
        # Precompute coverage for candidates
        # We only care about intersection with target_set
        cand_cover = []
        valid_cands = []
        for bi in candidates:
            inter = [ti for ti in block_cover[bi] if ti in target_set]
            if inter:
                cand_cover.append(inter)
                valid_cands.append(bi)
        
        if not valid_cands:
            return None # Should not happen if valid problem
            
        # If a single candidate covers everything, return it
        for i, cov in enumerate(cand_cover):
            if len(cov) == len(target_set):
                return [valid_cands[i]]
                
        best_solution = None
        best_size = len(valid_cands) + 1
        
        # Sort candidates by coverage size descending to find good solutions faster
        order = sorted(range(len(valid_cands)), key=lambda i: -len(cand_cover[i]))
        
        def solve(idx, current_covered, current_set):
            nonlocal best_solution, best_size
            
            if not target_set - current_covered:
                if len(current_set) < best_size:
                    best_solution = list(current_set)
                    best_size = len(current_set)
                return
                
            # Pruning: if current size + lower bound >= best_size, prune
            # Lower bound: ceil(remaining / max_coverage_of_any_remaining_candidate)
            # Approximate lower bound: 1 if any remaining
            if len(current_set) >= best_size:
                return
                
            if idx >= len(order):
                return
                
            bi = valid_cands[order[idx]]
            cov = set(cand_cover[order[idx]])
            
            # Option 1: Include this block
            new_covered = current_covered | cov
            current_set.append(bi)
            solve(idx + 1, new_covered, current_set)
            current_set.pop()
            
            # Option 2: Exclude this block
            # Optimization: If this block covers a unique element (in target_set not covered by others), we must include it.
            # For simplicity in this small context, we just recurse.
            # To speed up, we can skip if this block is "dominated" by another? 
            # Let's stick to simple B&B.
            solve(idx + 1, current_covered, current_set)

        solve(0, set(), [])
        
        if best_solution is None:
            # Fallback to greedy if B&B fails or times out (shouldn't happen for small sets)
            return greedy_cover_set(valid_cands, target_set)
            
        return best_solution

    def greedy_cover_set(candidates, target_set):
        """Greedy fallback for min set cover"""
        uncovered = set(target_set)
        selected = []
        cands = list(candidates)
        while uncovered and cands:
            best_b = -1
            max_cov = -1
            for bi in cands:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            selected.append(best_b)
            cands.remove(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return selected

    def prune_redundant(blocks):
        """Remove blocks that are not the sole cover for any t-subset."""
        if not blocks:
            return blocks
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    def greedy_cover_full():
        """Standard greedy set cover to get a valid initial solution."""
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_idx = next(iter(uncovered))
            candidates = t_to_blocks[seed_idx]
            best_b = -1
            max_cov = -1
            for bi in candidates:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return blocks

    # 1. Initial Solution
    init_blocks = greedy_cover_full()
    init_blocks = prune_redundant(init_blocks)
    
    best_blocks = list(init_blocks)
    best_score = len(best_blocks)
    
    rng = random.Random(42)
    
    # 2. Variable Neighborhood Search (VNS)
    # Structure:
    # - Maintain current set of blocks.
    # - Neighborhood k: randomly remove k blocks, find exact min cover for the gap.
    # - If the new set is smaller (or equal with diversity), accept.
    
    current_blocks = list(best_blocks)
    
    # Precompute in_set and cnt for current state
    in_set = np.zeros(B, dtype=bool)
    cnt = np.zeros(N, dtype=np.int32)
    for bi in current_blocks:
        in_set[bi] = True
        for ti in block_cover[bi]:
            cnt[ti] += 1
            
    # VNS Parameters
    max_k = 5 # Max number of blocks to swap out
    iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline - 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Choose neighborhood size k
        # Start small, increase if stuck? Or random?
        # Let's try random k from 1 to max_k
        k_swap = rng.randint(1, max_k)
        
        if len(current_blocks) < k_swap:
            continue
            
        # Select k blocks to remove
        # Random selection
        remove_indices = rng.sample(range(len(current_blocks)), k_swap)
        remove_blocks = [current_blocks[i] for i in remove_indices]
        
        # Calculate uncovered sets
        # We need to simulate removal
        temp_cnt = cnt.copy()
        for bi in remove_blocks:
            for ti in block_cover[bi]:
                temp_cnt[ti] -= 1
                
        uncovered = [ti for ti in range(N) if temp_cnt[ti] == 0]
        
        if not uncovered:
            # We removed blocks and covered everything? Impossible unless k=0 or redundant.
            # If redundant, we can just accept the removal.
            # But prune_redundant should have handled pure redundancies.
            # This implies the removed blocks were covering things that are still covered by others?
            # Wait, if temp_cnt[ti] == 0, it means it was ONLY covered by the removed blocks.
            # So if uncovered is empty, it means no t-subset lost its last cover.
            # This means the removed blocks were redundant.
            # We can remove them.
            new_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_indices]
            
            # Update state
            current_blocks = new_blocks
            in_set = np.zeros(B, dtype=bool)
            cnt = np.zeros(N, dtype=np.int32)
            for bi in current_blocks:
                in_set[bi] = True
                for ti in block_cover[bi]:
                    cnt[ti] += 1
                    
            if len(current_blocks) < best_score:
                best_score = len(current_blocks)
                best_blocks = list(current_blocks)
            continue
            
        # We have a gap: `uncovered`
        # Find candidates to cover the gap
        # Candidates are blocks NOT in current_blocks (after removal)
        # Actually, we can use ANY block not in the final set.
        # Let's gather candidates from t_to_blocks for each uncovered subset.
        
        candidate_set = set()
        for ti in uncovered:
            for bi in t_to_blocks[ti]:
                # Exclude blocks we are removing?
                # We are replacing remove_blocks.
                # So we can use any block not in (current_blocks - remove_blocks).
                # But simpler: just use any block not in current_blocks?
                # No, we are removing remove_blocks, so they are available again.
                # But we want to find a NEW set.
                # Let's consider all blocks not in the "kept" set.
                # Kept set = current_blocks - remove_blocks
                pass
        
        # Efficient candidate generation:
        # For each uncovered ti, add all blocks covering it.
        # Then filter out blocks that are in the "kept" part of current_blocks.
        
        kept_blocks = set(current_blocks) - set(remove_blocks)
        
        for ti in uncovered:
            for bi in t_to_blocks[ti]:
                if bi not in kept_blocks:
                    candidate_set.add(bi)
        
        if not candidate_set:
            continue
            
        # Solve Min Set Cover on candidate_set for `uncovered`
        # To make B&B fast, we might want to reduce candidate_set.
        # Dominance: If block A covers all of block B's intersection with uncovered, B is dominated.
        # This is expensive. Let's just pass a subset if candidate_set is too large.
        
        candidates_list = list(candidate_set)
        
        # If candidates are too many for B&B (e.g., > 50), use Greedy + Local Search or just Greedy
        # B&B is exponential.
        # For k_swap=1, uncovered is small (<=56). Candidates might be ~100-200.
        # B&B on 200 items is too slow.
        # Strategy:
        # 1. Use Greedy to get an initial cover R.
        # 2. Use B&B on the "relevant" candidates to see if we can do better than |R|?
        # Or just use Greedy, but with a twist:
        # The "Exact" part is key. Let's try to limit candidates.
        
        # Heuristic: Only consider candidates that cover at least 2 uncovered subsets?
        # Or just take the top M candidates by coverage of uncovered.
        
        uncovered_set = set(uncovered)
        
        # Score candidates by coverage of uncovered_set
        scored_cands = []
        for bi in candidates_list:
            cov = 0
            for ti in block_cover[bi]:
                if ti in uncovered_set:
                    cov += 1
            if cov > 0:
                scored_cands.append((cov, bi))
                
        scored_cands.sort(reverse=True)
        
        # Limit to top 30 candidates for B&B
        top_cands = [bi for _, bi in scored_cands[:30]]
        
        if not top_cands:
            continue
            
        # Try Exact B&B on top_cands
        # If it fails (timeout or no solution), fall back to Greedy on all candidates
        # We need a timeout for B&B.
        
        best_repair = None
        best_repair_size = len(current_blocks) + 1 # Worse than doing nothing
        
        # Run B&B
        # To avoid hanging, we'll add a simple step counter in the B&B recursion?
        # Python recursion limit and speed...
        # Let's implement a time-checked B&B.
        
        bnb_steps = 0
        bnb_time_limit = 0.5 # 0.5s per B&B call
        
        def bnb_solve(cands, target):
            nonlocal bnb_steps, bnb_time_limit
            if not target:
                return []
            
            # Precompute
            cand_cover = []
            valid_cands = []
            for bi in cands:
                inter = [ti for ti in block_cover[bi] if ti in target]
                if inter:
                    cand_cover.append(inter)
                    valid_cands.append(bi)
                    
            if not valid_cands:
                return None
                
            for i, cov in enumerate(cand_cover):
                if len(cov) == len(target):
                    return [valid_cands[i]]
                    
            best_sol = None
            best_sz = len(valid_cands) + 1
            order = sorted(range(len(valid_cands)), key=lambda i: -len(cand_cover[i]))
            
            def rec(idx, covered, sol):
                nonlocal best_sol, best_sz, bnb_steps
                bnb_steps += 1
                if bnb_steps % 1000 == 0 and time.time() > t0 + bnb_time_limit:
                    return True # Timeout signal
                if not target - covered:
                    if len(sol) < best_sz:
                        best_sol = list(sol)
                        best_sz = len(sol)
                    return False
                if len(sol) >= best_sz:
                    return False
                if idx >= len(order):
                    return False
                if time.time() > t0 + bnb_time_limit:
                    return True
                    
                bi = valid_cands[order[idx]]
                cov = set(cand_cover[order[idx]])
                
                sol.append(bi)
                if rec(idx + 1, covered | cov, sol):
                    return True
                sol.pop()
                
                if rec(idx + 1, covered, sol):
                    return True
                return False
                
            timed_out = rec(0, set(), [])
            if timed_out:
                return None
            return best_sol

        repair = bnb_solve(top_cands, uncovered_set)
        
        if repair is None:
            # Fallback to Greedy on all candidates
            repair = greedy_cover_set(candidates_list, uncovered_set)
            if not repair:
                continue
            
        # Construct new block set
        # New set = Kept Blocks + Repair
        new_blocks_set = kept_blocks.union(repair)
        
        # Check for duplicates? Sets handle it.
        # But we need to ensure distinct blocks.
        
        new_blocks = list(new_blocks_set)
        
        if len(new_blocks) < len(current_blocks):
            # Accept
            current_blocks = new_blocks
            # Update state
            in_set = np.zeros(B, dtype=bool)
            cnt = np.zeros(N, dtype=np.int32)
            for bi in current_blocks:
                in_set[bi] = True
                for ti in block_cover[bi]:
                    cnt[ti] += 1
                    
            if len(current_blocks) < best_score:
                best_score = len(current_blocks)
                best_blocks = list(current_blocks)
                
        elif len(new_blocks) == len(current_blocks):
            # Accept with some probability to escape local minima
            if rng.random() < 0.2:
                current_blocks = new_blocks
                in_set = np.zeros(B, dtype=bool)
                cnt = np.zeros(N, dtype=np.int32)
                for bi in current_blocks:
                    in_set[bi] = True
                    for ti in block_cover[bi]:
                        cnt[ti] += 1

    # Final Prune
    best_blocks = prune_redundant(best_blocks)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            cnt_final[bi] += 1 # Wait, bug in verification? No, cnt_final[ti]
            
    # Correct verification
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in greedy_cover_full()]
        
    return [list(all_blocks[bi]) for bi in best_blocks]
