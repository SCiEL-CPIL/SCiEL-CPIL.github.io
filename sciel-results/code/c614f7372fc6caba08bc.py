import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # ---------- Precomputation ----------
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    block_idx_map = {b: i for i, b in enumerate(all_k_blocks)}

    # For each block, the list of t-subset indices it covers
    block_ts = []
    # For each t-subset, list of block indices that cover it
    ts_blocks = [[] for _ in range(num_t)]
    for b in all_k_blocks:
        lst = [ts_index[ts] for ts in itertools.combinations(b, t)]
        block_ts.append(lst)
        for idx in lst:
            ts_blocks[idx].append(block_idx_map[b])

    def C(n, r):
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    def is_valid(blocks):
        """Exhaustive check that every t-subset is covered."""
        covered = np.zeros(num_t, dtype=bool)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered[ts_index[ts]] = True
        return bool(np.all(covered))

    def greedy_construct(seed, top_k=5):
        """Standard greedy set cover with random tie-breaking."""
        random.seed(seed)
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        uncovered = np.full(num_t, True, dtype=bool)
        selected = np.full(num_k, False, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 3.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            pk = min(top_k, len(cands))
            if pk < len(cands):
                chosen = random.sample(list(cands), pk)
                b = chosen[random.randint(0, pk - 1)]
            else:
                b = cands[random.randint(0, len(cands) - 1)]
            sol.append(list(all_k_blocks[b]))
            selected[b] = True
            gain[b] = -1
            for idx in block_ts[b]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    for nb in ts_blocks[idx]:
                        if not selected[nb]:
                            gain[nb] -= 1
        return sol

    def thin(blocks, target):
        """Randomly keep a subset of blocks of size target (or all if fewer)."""
        if len(blocks) <= target:
            return list(blocks)
        return [blocks[i] for i in random.sample(range(len(blocks)), target)]

    def repair(blocks, seed):
        """Greedily add blocks to cover all t-subsets not yet covered by `blocks`."""
        random.seed(seed)
        covered = np.zeros(num_t, dtype=bool)
        for b in blocks:
            for idx in block_ts[block_idx_map[tuple(b)]]:
                covered[idx] = True

        # Gain for each candidate block = number of currently-uncovered t-subsets it covers
        gain = np.zeros(num_k, dtype=np.int32)
        for j in range(num_k):
            g = 0
            for idx in block_ts[j]:
                if not covered[idx]:
                    g += 1
            gain[j] = g

        in_sol = set(tuple(b) for b in blocks)
        sol = list(blocks)
        while True:
            if time.time() > deadline - 3.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = [int(c) for c in np.where(gain == mg)[0] if tuple(all_k_blocks[c]) not in in_sol]
            if not cands:
                break
            pk = min(5, len(cands))
            if pk < len(cands):
                chosen = random.sample(cands, pk)
                b = chosen[random.randint(0, pk - 1)]
            else:
                b = cands[random.randint(0, len(cands) - 1)]
            sol.append(list(all_k_blocks[b]))
            in_sol.add(tuple(all_k_blocks[b]))
            for idx in block_ts[b]:
                if not covered[idx]:
                    covered[idx] = True
                    for nb in ts_blocks[idx]:
                        if tuple(all_k_blocks[nb]) not in in_sol:
                            gain[nb] -= 1
        return sol

    def iterative_removal(blocks):
        """Remove blocks that are fully redundant; repeat until stable."""
        if not blocks:
            return blocks
        # Map each block to its index for fast lookup
        # Build cover count
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for idx in block_ts[block_idx_map[tuple(b)]]:
                cover_count[idx] += 1

        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                redundant = True
                for idx in block_ts[block_idx_map[tuple(blocks[i])]]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_ts[block_idx_map[tuple(blocks[i])]]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # ---------- Main search ----------
    best_solution = None
    best_len = float('inf')

    # Phase 1: Build an initial covering via a few greedy runs
    for seed in range(10):
        if time.time() > deadline - 20.0:
            break
        sol = greedy_construct(seed, top_k=random.choice([3, 5, 10]))
        if sol is not None:
            sol = iterative_removal(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    if best_solution is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        uncovered = np.full(num_t, True, dtype=bool)
        selected = np.full(num_k, False, dtype=bool)
        sol = []
        while True:
            mg = int(np.max(gain))
            if mg <= 0:
                break
            b = int(np.argmax(gain))
            sol.append(list(all_k_blocks[b]))
            selected[b] = True
            gain[b] = -1
            for idx in block_ts[b]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    for nb in ts_blocks[idx]:
                        if not selected[nb]:
                            gain[nb] -= 1
        best_solution = sol
        best_len = len(sol)

    # Phase 2: Iterative compression
    # Start from best_len and try to go lower
    current_target = best_len - 1
    attempt_seed = 100
    while current_target >= max(264, int(0.9 * best_len)) and time.time() < deadline - 10.0:
        improved_this_level = False
        # Try multiple thinning+repair attempts at this target
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            thinned = thin(best_solution, current_target)
            rep = repair(thinned, seed=attempt_seed + attempt)
            attempt_seed += 1
            if rep is None:
                continue
            rep = iterative_removal(rep)
            if len(rep) < best_len:
                best_solution = rep
                best_len = len(rep)
                improved_this_level = True
                # If we improved, we can try even lower targets immediately
                current_target = best_len - 1
                break
        if not improved_this_level:
            # No improvement at this target; try a slightly larger target (less aggressive thinning)
            # or move to a lower target? Standard iterative compression: if target fails, increase target.
            # But here we want to minimize, so if we can't beat best_len with target=current_target,
            # maybe try a target between current_target and best_len? 
            # Actually, if thinning to current_target and repairing doesn't beat best_len, 
            # increasing the target (keeping more blocks) makes repair easier but final size likely larger.
            # Decreasing the target makes repair harder.
            # So if current_target didn't work, we should probably try a target closer to best_len (less thinning) 
            # but that's what we just did (current_target = best_len - 1).
            # If that didn't work, we might try best_len - 2? No, that's harder.
            # Wait, my loop logic: current_target starts at best_len - 1.
            # If it fails, we should probably give up on this specific target and try a different randomization?
            # Or accept that we might not find an improvement at this specific target and try a slightly higher target?
            # No, higher target means we keep more blocks initially, so repair adds fewer, final size likely >= current_target.
            # If current_target < best_len, and we fail to find a solution < best_len, we can try current_target+1?
            # But current_target+1 is closer to best_len, so more likely to succeed but might not beat best_len.
            # Actually, the goal is to find ANY solution < best_len.
            # So we should try targets from best_len-1 downwards? No, downwards is harder.
            # We should try targets from best_len-1 upwards? No, upwards is > best_len-1, so might not beat best_len.
            # The only way to beat best_len is to find a solution of size <= best_len - 1.
            # So we must try targets <= best_len - 1.
            # If target = best_len - 1 fails, maybe target = best_len - 2 will succeed? Unlikely, it's harder.
            # So if best_len - 1 fails, we are stuck? 
            # Unless the repair process finds a solution smaller than the target? No, repair only adds blocks.
            # So the final size is at least the target size (after thinning) minus any removals.
            # So if we thin to T, final size >= T (minus removals).
            # To beat best_len, we need final size <= best_len - 1.
            # So we need T <= best_len - 1.
            # If T = best_len - 1 fails to yield a valid solution < best_len, maybe a different random thinning/repair will?
            # So we should keep trying T = best_len - 1 with different seeds.
            # My loop above tries 20 attempts at the same target. If all fail, we break.
            # Should we then try T = best_len - 2? It's harder, so less likely.
            # So it's better to keep trying T = best_len - 1 with more seeds?
            # But we have a time limit.
            # Let's just break out if we can't improve at the current target after a few tries, 
            # and try a slightly higher target? No, that won't beat best_len.
            # So we are stuck at best_len. 
            # Maybe we should try local search instead?
            # For now, let's just break out of the while loop if we can't improve at the current target.
            break

    # Phase 3: Final cleanup
    best_solution = iterative_removal(best_solution)
    
    # Final validation (optional, but good for safety)
    # if not is_valid(best_solution):
    #     # Fallback to a known valid solution if something went wrong
    #     sol = greedy_construct(0, top_k=5)
    #     if sol is not None:
    #         best_solution = sol

    return best_solution
