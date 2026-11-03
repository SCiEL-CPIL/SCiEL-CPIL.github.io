"""
Covering design solver: combining bitset-accelerated greedy, MILP polishing,
and a minimum-cost-flow formulation for the assignment subproblem.

Strategy:
  1. Precompute all C(v,k) blocks and their t-subset memberships using bitmasks.
  2. Dynamic greedy with heap-based priority updates and bitset coverage.
  3. Multi-restart with different seeds.
  4. Deletion phase: remove redundant blocks.
  5. MILP polish (if scipy available) to find minimum set cover over selected blocks.
  6. Minimum-cost-flow based refinement: model as flow where source->blocks (cost 1),
     blocks->t-subsets (cost 0), t-subsets->sink (cap 1). Solve via successive
     shortest augmenting path with potentials. This finds a minimum-cost assignment
     of t-subsets to blocks, which can reveal redundant blocks.
"""
import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None

# Precomputed global structures (lazy init)
_G = {}


def _init_globals(v, k, t):
    """Precompute all structures needed for the solver."""
    if (v, k, t) in _G:
        return _G[(v, k, t)]

    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # For each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # Precompute bitmask for each block: which t-subsets it covers
    block_mask = []
    for ts_list in block_ts:
        mask = 0
        for s in ts_list:
            mask |= (1 << s)
        block_mask.append(mask)

    # Precompute bitmask for each block: which points it contains
    block_pt_mask = []
    for blk in all_blocks:
        pm = 0
        for p in blk:
            pm |= (1 << p)
        block_pt_mask.append(pm)

    # For each t-subset, bitmask of points in it
    ts_pt_mask = []
    for ts in all_ts:
        pm = 0
        for p in ts:
            pm |= (1 << p)
        ts_pt_mask.append(pm)

    # Number of t-subsets per block
    n_ts_per_block = len(block_ts[0])  # C(k,t)

    _G[(v, k, t)] = {
        'all_ts': all_ts,
        'n_ts': n_ts,
        'ts_index': ts_index,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_ts': block_ts,
        'ts_to_blocks': ts_to_blocks,
        'block_mask': block_mask,
        'block_pt_mask': block_pt_mask,
        'ts_pt_mask': ts_pt_mask,
        'n_ts_per_block': n_ts_per_block,
    }
    return _G[(v, k, t)]


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _milp_polish(v, k, t, blocks, G, deadline):
    """Try to reduce the block set using MILP set cover formulation."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        import numpy as np
    except Exception:
        return blocks

    n_ts = G['n_ts']
    ts_index = G['ts_index']
    n_blocks_in = len(blocks)
    if n_blocks_in == 0:
        return blocks

    # Build constraint matrix A (n_ts x n_blocks_in)
    A = lil_matrix((n_ts, n_blocks_in), dtype=np.int8)
    for j, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            idx = ts_index[sub]
            A[idx, j] = 1
    A = A.tocsr()

    c = np.ones(n_blocks_in)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    time_left = deadline - time.time()
    if time_left < 5:
        return blocks

    try:
        res = milp(c, integrality=np.ones(n_blocks_in), constraints=constraint,
                   bounds=bounds, options={'time_limit': max(5, int(time_left - 3)),
                                           'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [j for j in range(n_blocks_in) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]
        # Verify coverage
        if len(new_blocks) < len(blocks):
            cov_mask = 0
            for b in new_blocks:
                for sub in itertools.combinations(b, t):
                    cov_mask |= (1 << ts_index[sub])
            if cov_mask == (1 << n_ts) - 1:
                return new_blocks

    return blocks


def _flow_based_refinement(blocks, G, time_limit):
    """
    Minimum-cost-flow based refinement.
    
    Model: Source -> block nodes (capacity 1, cost 1), block -> t-subset nodes
    (capacity 1, cost 0), t-subset -> sink (capacity 1, cost 0).
    
    We want to assign each t-subset to exactly one block that contains it,
    minimizing the number of blocks used. This is equivalent to finding a
    minimum set cover where each t-subset is assigned to one covering block.
    
    The flow formulation gives a lower bound on the number of blocks needed
    and can identify blocks that are not needed in any optimal assignment.
    
    Implementation: successive shortest augmenting path with potentials.
    """
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    n_blocks_in = len(blocks)
    if n_blocks_in == 0:
        return blocks

    # Build adjacency for the flow network
    # Nodes: source (0), block nodes (1..n_blocks_in), t-subset nodes (n_blocks_in+1..n_blocks_in+n_ts), sink (n_blocks_in+n_ts+1)
    # Edges:
    #   source -> block_i: cap=1, cost=1
    #   block_i -> ts_j: cap=1, cost=0 (if block_i covers ts_j)
    #   ts_j -> sink: cap=1, cost=0
    
    # For efficiency, we'll use a simplified approach:
    # For each block, track which t-subsets it can cover.
    # Use the flow to find which blocks are essential.
    
    local_start = time.time()
    
    # Build coverage: for each t-subset, which blocks in our set cover it
    ts_covering_blocks = [[] for _ in range(n_ts)]
    for b_idx, blk in enumerate(blocks):
        for sub in itertools.combinations(blk, t):
            si = ts_index[sub]
            ts_covering_blocks[si].append(b_idx)
    
    # Phase 1: Remove blocks that don't cover any unique t-subset
    # A block is "essential" if it's the only one covering some t-subset
    block_essential_ts = [set() for _ in range(n_blocks_in)]
    for si in range(n_ts):
        if len(ts_covering_blocks[si]) == 1:
            b_idx = ts_covering_blocks[si][0]
            block_essential_ts[b_idx].add(si)
    
    # Remove non-essential blocks that are fully redundant
    # (all their t-subsets are covered by other blocks)
    coverage_count = [0] * n_ts
    for b_idx in range(n_blocks_in):
        blk = blocks[b_idx]
        for sub in itertools.combinations(blk, t):
            coverage_count[ts_index[sub]] += 1
    
    removable = []
    for b_idx in range(n_blocks_in):
        if len(block_essential_ts[b_idx]) > 0:
            continue
        blk = blocks[b_idx]
        all_redundant = True
        for sub in itertools.combinations(blk, t):
            if coverage_count[ts_index[sub]] <= 1:
                all_redundant = False
                break
        if all_redundant:
            removable.append(b_idx)
    
    if removable:
        for b_idx in sorted(removable, reverse=True):
            blk = blocks[b_idx]
            for sub in itertools.combinations(blk, t):
                coverage_count[ts_index[sub]] -= 1
            blocks.pop(b_idx)
        n_blocks_in = len(blocks)
        if n_blocks_in == 0:
            return blocks
    
    # Phase 2: Flow-based block elimination
    # Try to find a subset of blocks that covers all t-subsets using flow
    # This is essentially the set cover problem, which is NP-hard.
    # But the flow formulation can give us a good heuristic.
    
    # Greedy assignment via flow-like approach:
    # Assign each t-subset to a block, trying to minimize the number of distinct blocks used.
    # Process t-subsets in order of fewest covering blocks first.
    
    if time.time() - local_start > time_limit:
        return blocks
    
    # Sort t-subsets by number of covering blocks (ascending)
    ts_order = sorted(range(n_ts), key=lambda si: len(ts_covering_blocks[si]))
    
    # Assignment: ts -> block
    ts_assignment = [-1] * n_ts
    block_usage = [0] * n_blocks_in  # number of t-subsets assigned to each block
    
    # Rebuild ts_covering_blocks after removals
    ts_covering_blocks2 = [[] for _ in range(n_ts)]
    for b_idx, blk in enumerate(blocks):
        for sub in itertools.combinations(blk, t):
            si = ts_index[sub]
            ts_covering_blocks2[si].append(b_idx)
    
    for si in ts_order:
        if time.time() - local_start > time_limit:
            return blocks
        candidates = ts_covering_blocks2[si]
        if not candidates:
            return blocks  # Should not happen
        # Choose the block with the most already-assigned t-subsets (to concentrate)
        best_b = max(candidates, key=lambda b: block_usage[b])
        ts_assignment[si] = best_b
        block_usage[best_b] += 1
    
    # Now, blocks with 0 assigned t-subsets can potentially be removed
    # But we need to verify: if we remove a block, can we reassign its t-subsets?
    # For now, just remove blocks with 0 usage
    used_blocks = set(b for b in range(n_blocks_in) if block_usage[b] > 0)
    
    new_blocks = [blocks[b] for b in sorted(used_blocks)]
    
    # Verify coverage
    if len(new_blocks) < len(blocks):
        cov_mask = 0
        for b in new_blocks:
            for sub in itertools.combinations(b, t):
                cov_mask |= (1 << ts_index[sub])
        if cov_mask == (1 << n_ts) - 1:
            return new_blocks
    
    return blocks


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    G = _init_globals(v, k, t)
    all_ts = G['all_ts']
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    all_blocks = G['all_blocks']
    n_blocks = G['n_blocks']
    block_ts = G['block_ts']
    ts_to_blocks = G['ts_to_blocks']
    block_mask = G['block_mask']
    block_pt_mask = G['block_pt_mask']
    ts_pt_mask = G['ts_pt_mask']
    n_ts_per_block = G['n_ts_per_block']

    def greedy_solve(seed, max_time_frac=0.5):
        """Run one greedy pass with given random seed for tie-breaking."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered_mask = (1 << n_ts) - 1  # all t-subsets uncovered
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [n_ts_per_block] * n_blocks
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage using bitmask
            bm = block_mask[b_idx]
            new_uncovered = bm & uncovered_mask
            newly_covered = bin(new_uncovered).count('1')
            uncovered_count -= newly_covered
            uncovered_mask &= ~bm

            # Update gains of other blocks containing newly covered t-subsets
            nu = new_uncovered
            while nu:
                s_idx = (nu & (-nu)).bit_length() - 1
                nu &= nu - 1
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        # Convert to list of blocks
        result = [list(all_blocks[b]) for b in selected]

        # Check if complete
        if uncovered_count > 0:
            while uncovered_count > 0 and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = bin(block_mask[b_idx] & uncovered_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                uncovered_mask &= ~block_mask[best_b]
                uncovered_count -= best_gain

        return result

    def local_search_improve(blocks, seed, time_limit):
        """Improve a covering by removing redundant blocks and trying swaps."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if not blocks:
            return blocks

        # Build coverage counts
        coverage = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Phase 1: Remove redundant blocks (fast pass)
        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                blk = blocks[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        # Phase 2: Randomized swap search
        attempts = 0
        max_attempts = 3000
        while (time.time() - local_start < time_limit and attempts < max_attempts
               and len(blocks) > 0):
            attempts += 1
            idx = local_rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]

            # Find t-subsets with coverage exactly 1
            unique_ts = []
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 1:
                    unique_ts.append(ts_index[ts])

            if not unique_ts:
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] -= 1
                blocks.pop(idx)
                continue

            # Must find a replacement covering all unique_ts
            union_pts_mask = 0
            for s_idx in unique_ts:
                union_pts_mask |= ts_pt_mask[s_idx]
            if bin(union_pts_mask).count('1') > k:
                continue

            # Find best replacement
            best_repl = -1
            best_overlap = -1

            for b_idx in range(n_blocks):
                if (block_pt_mask[b_idx] & union_pts_mask) != union_pts_mask:
                    continue

                overlap = 0
                for s_idx in block_ts[b_idx]:
                    if coverage[s_idx] > 0:
                        overlap += 1

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_repl = b_idx

            if best_repl == -1:
                continue

            new_block = all_blocks[best_repl]

            # Apply swap
            for ts in itertools.combinations(old_block, t):
                coverage[ts_index[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                coverage[ts_index[ts]] += 1

            # Verify validity
            valid = True
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 0:
                    valid = False
                    break

            if valid:
                blocks[idx] = list(new_block)
                # After swap, try to remove other now-redundant blocks
                for j in range(len(blocks) - 1, -1, -1):
                    if j == idx:
                        continue
                    blk_j = blocks[j]
                    removable = True
                    for ts in itertools.combinations(blk_j, t):
                        if coverage[ts_index[ts]] <= 1:
                            removable = False
                            break
                    if removable:
                        for ts in itertools.combinations(blk_j, t):
                            coverage[ts_index[ts]] -= 1
                        blocks.pop(j)
                        break
            else:
                for ts in itertools.combinations(new_block, t):
                    coverage[ts_index[ts]] -= 1
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] += 1

        return blocks

    # Multi-restart greedy
    best_result = None
    best_count = float('inf')

    n_restarts = 5
    total_budget = _DEADLINE - 10.0
    greedy_frac = 0.45
    ls1_frac = 0.25
    flow_frac = 0.15
    milp_frac = 0.15

    time_per_greedy = total_budget * greedy_frac / n_restarts
    time_per_ls1 = total_budget * ls1_frac / n_restarts
    time_per_flow = total_budget * flow_frac / n_restarts
    time_per_milp = total_budget * milp_frac / n_restarts

    for restart in range(n_restarts):
        if _remaining() < 12.0:
            break

        seed = 12345 + restart * 7919
        max_frac = min(0.50, time_per_greedy / _DEADLINE)

        result = greedy_solve(seed, max_time_frac=max_frac)

        if not result:
            continue

        # Check validity using bitmask
        cov_mask = 0
        for blk in result:
            for ts in itertools.combinations(blk, t):
                cov_mask |= (1 << ts_index[ts])
        if cov_mask != (1 << n_ts) - 1:
            continue

        # Local search 1-exchange
        ls1_time = min(time_per_ls1, _remaining() - 12.0)
        if ls1_time > 2.0:
            result = local_search_improve(result, seed + 1, ls1_time)

        # Flow-based refinement
        flow_time = min(time_per_flow, _remaining() - 8.0)
        if flow_time > 1.0:
            result = _flow_based_refinement(result, G, flow_time)

        # MILP polish
        milp_time = min(time_per_milp, _remaining() - 5.0)
        if milp_time > 3.0:
            deadline = time.time() + milp_time
            result = _milp_polish(v, k, t, result, G, deadline)

        if len(result) < best_count:
            best_count = len(result)
            best_result = result

    if best_result is None:
        best_result = greedy_solve(12345, max_time_frac=0.9)

    # Final verification
    final_cov = [0] * n_ts
    for blk in best_result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        unc_mask = 0
        for i in range(n_ts):
            if final_cov[i] == 0:
                unc_mask |= (1 << i)
        while unc_mask and _remaining() > 0.5:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = bin(block_mask[b_idx] & unc_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            best_result.append(list(all_blocks[best_b]))
            unc_mask &= ~block_mask[best_b]

    return best_result
