"""Covering design C(v,k,t) — greedy + bitmask + redundancy removal."""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets and map them to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubsets)
    # Map each t-subset (tuple) to an integer index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each point, precompute the list of t-subset indices it participates in
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubsets:
        for p in ts:
            point_tsubs[p].append(tsub_to_idx[ts])

    def block_mask(block):
        """Compute a numpy bool array of length n_tsubs: True if t-subset covered by block."""
        mask = np.zeros(n_tsubs, dtype=np.bool_)
        for ts in itertools.combinations(block, t):
            mask[tsub_to_idx[ts]] = True
        return mask

    def greedy_cover(seed=None, time_limit=60.0):
        """Randomized greedy set cover with bitmask."""
        if seed is not None:
            random.seed(seed)
        uncovered = np.ones(n_tsubs, dtype=np.bool_)
        blocks = []
        while uncovered.any():
            if time.time() > time_limit:
                break
            # Pick a random uncovered t-subset to start
            uncov_indices = np.nonzero(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            start_ts_idx = uncov_indices[random.randrange(len(uncov_indices))]
            start_ts = all_tsubsets[start_ts_idx]
            block = list(start_ts)
            # Add points greedily to maximize coverage
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for ts_idx in point_tsubs[x]:
                        if uncovered[ts_idx]:
                            # Check if this t-subset would be in the block
                            ts = all_tsubsets[ts_idx]
                            if set(ts) <= set(block) | {x}:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    # Pick random
                    available = [x for x in range(v) if x not in block]
                    best_x = random.choice(available)
                block.append(best_x)
            block.sort()
            m = block_mask(block)
            blocks.append(block)
            uncovered &= ~m
        return blocks

    def redundancy_removal(blocks, time_limit=5.0):
        """Remove redundant blocks (those whose t-subsets are all covered by others)."""
        if not blocks:
            return blocks
        # Build coverage count per t-subset
        counts = np.zeros(n_tsubs, dtype=np.int32)
        block_masks = []
        for b in blocks:
            m = block_mask(b)
            block_masks.append(m)
            counts += m.astype(np.int32)
        
        changed = True
        remaining = list(range(len(blocks)))
        remaining_masks = block_masks[:]
        while changed and time.time() < time_limit:
            changed = False
            new_remaining = []
            new_masks = []
            for i in remaining:
                m = remaining_masks[i]
                # Check if all t-subsets in this block are covered by other remaining blocks
                covered_by_others = np.zeros(n_tsubs, dtype=np.int32)
                for j in remaining:
                    if j != i:
                        covered_by_others += remaining_masks[j].astype(np.int32)
                if np.all(covered_by_others[m] >= 1):
                    changed = True
                else:
                    new_remaining.append(i)
                    new_masks.append(m)
            remaining = new_remaining
            remaining_masks = new_masks
        return [blocks[i] for i in remaining]

    def local_improvement(blocks, time_limit=30.0):
        """Try to swap blocks to reduce count. Replace a block with a better one."""
        if not blocks:
            return blocks
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            m = block_mask(b)
            counts += m.astype(np.int32)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx in range(len(blocks)):
                if time.time() > time_limit:
                    break
                # Temporarily remove block idx
                m_idx = block_mask(blocks[idx])
                counts -= m_idx.astype(np.int32)
                # Find uncovered t-subsets
                uncov = np.nonzero(counts == 0)[0]
                if len(uncov) == 0:
                    # Block is redundant, remove it
                    blocks.pop(idx)
                    improved = True
                    break
                # Try to find a better replacement
                best_new_block = None
                best_gain = -1
                # Sample some candidate blocks
                candidates = set()
                for u_idx in uncov[:5]:
                    ts = all_tsubsets[u_idx]
                    # Try blocks containing this t-subset with random other points
                    base = list(ts)
                    for _ in range(20):
                        extra = random.sample([x for x in range(v) if x not in base], k - t)
                        cand = sorted(base + extra)
                        candidates.add(tuple(cand))
                for cand in list(candidates)[:50]:
                    gain = 0
                    for ts_idx in point_tsubs[cand[0]]:
                        if counts[ts_idx] == 0:
                            # Check if this t-subset is in cand
                            ts = all_tsubsets[ts_idx]
                            if set(ts) <= set(cand):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new_block = list(cand)
                if best_new_block is not None and best_gain > 0:
                    blocks[idx] = best_new_block
                    m_new = block_mask(best_new_block)
                    counts += m_new.astype(np.int32)
                    improved = True
                else:
                    # Restore
                    counts += m_idx.astype(np.int32)
        return blocks

    # Phase 1: Multiple greedy runs, keep best
    best_blocks = None
    best_count = float('inf')
    max_greedy_runs = 10
    for run in range(max_greedy_runs):
        if time.time() > DEADLINE - 60:
            break
        blocks = greedy_cover(seed=42 + run, time_limit=min(30.0, DEADLINE - time.time() - 40))
        if blocks:
            cnt = len(blocks)
            if cnt < best_count:
                best_count = cnt
                best_blocks = blocks[:]
    
    if best_blocks is None:
        # Fallback: basic greedy
        blocks = greedy_cover(seed=0, time_limit=DEADLINE - time.time() - 5)
        if blocks:
            best_blocks = blocks
        else:
            return []

    # Phase 2: Redundancy removal
    if time.time() < DEADLINE - 30:
        best_blocks = redundancy_removal(best_blocks, time_limit=min(10.0, DEADLINE - time.time() - 30))

    # Phase 3: Local improvement
    if time.time() < DEADLINE - 15:
        best_blocks = local_improvement(best_blocks, time_limit=min(20.0, DEADLINE - time.time() - 15))

    # Phase 4: Final redundancy pass
    if time.time() < DEADLINE - 5:
        best_blocks = redundancy_removal(best_blocks, time_limit=min(5.0, DEADLINE - time.time() - 5))

    return best_blocks
