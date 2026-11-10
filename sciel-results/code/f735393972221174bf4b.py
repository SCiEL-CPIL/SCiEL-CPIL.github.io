import itertools
import random
import time
import numpy as np


def _build_masks(v, k, t):
    """Precompute bitmask for each t-subset and for each point."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_mask = np.zeros(n_tsubs, dtype=np.uint64)
    for i, ts in enumerate(all_tsubs):
        m = 0
        for x in ts:
            m |= (1 << x)
        tsub_mask[i] = m
    point_mask = np.array([1 << x for x in range(v)], dtype=np.uint64)
    return all_tsubs, tsub_mask, point_mask


def _complement_set(v, k, block):
    """Set of points NOT in block (the '11' side)."""
    return set(range(v)) - set(block)


def _is_redundant(compl, tsub_mask, t, point_mask):
    """
    A block is redundant iff for every t-subset, it does NOT uniquely cover it.
    Block covers ts iff |ts ∩ compl| <= t-1, i.e. ts ∩ compl has size <= 4.
    A block covers ts iff (ts_mask & compl_mask) has popcount <= t-1.
    Redundant iff no t-subset has popcount(ts & compl) == t-1 AND is covered ONLY by this block.
    Simpler: block covers ts iff popcount(ts_mask & compl_mask) < t  (i.e. <= t-1).
    Wait: ts has t elements. ts ∩ block has size t - |ts ∩ compl|.
    ts ⊆ block iff ts ∩ compl = empty iff |ts ∩ compl| = 0.
    So block covers ts iff (ts_mask & compl_mask) == 0.
    """
    compl_mask = 0
    for x in compl:
        compl_mask |= (1 << x)
    # covered = t-subsets with (ts_mask & compl_mask) == 0
    inter = tsub_mask & compl_mask
    covered_mask = (inter == 0)
    return covered_mask


def _fast_greedy(v, k, t, rng, deadline, all_tsubs, tsub_mask, point_mask):
    """
    Greedy: repeatedly pick block maximizing newly covered t-subsets.
    A t-subset is covered by block B iff B ∩ ts = ts, i.e. ts ⊆ B.
    Equivalent: ts ∩ (V\B) = empty.
    """
    n_tsubs = len(all_tsubs)
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []

    # For each point x, which t-subsets contain x (as a boolean mask over t-subsets)
    # point_covers[x] = boolean array: tsub i has x in it
    point_covers = np.zeros((v, n_tsubs), dtype=bool)
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_covers[x, i] = True

    # For a partial block (set of points), the t-subsets it covers = intersection of point_covers[x] for x in block
    # That's expensive. Instead: a t-subset ts is covered by block B iff all t elements of ts are in B.
    # For a full block of size k: ts ⊆ B iff ts ∩ (V\B) = empty.
    # Precompute complement mask per block on the fly.

    while np.any(uncovered) and time.time() < deadline - 2.0:
        # Pick block: try to maximize coverage of uncovered t-subsets.
        # For a candidate block B, newly covered = uncovered t-subsets that are subsets of B.
        # ts ⊆ B iff (ts_mask & ~B_mask) == 0, i.e. (ts_mask & compl_mask) == 0 where compl = V\B.

        # Heuristic: start from a random uncovered t-subset, build block around it.
        uncovered_idx = np.where(uncovered)[0]
        start_i = int(rng.choice(uncovered_idx))
        start = all_tsubs[start_i]
        block = list(start)
        block_set = set(block)

        # Greedily add points to maximize uncovered t-subsets that become covered.
        # When we have partial block P (size s < k), adding point x makes P∪{x}.
        # A t-subset ts becomes covered by P∪{x} (and wasn't by P) iff:
        #   ts ⊆ P∪{x} and ts ⊄ P.
        # ts ⊆ P∪{x} iff ts \ {x} ⊆ P (when x in ts) or ts ⊆ P.
        # Since ts ⊄ P, we need x in ts and ts\{x} ⊆ P.
        # So gain(x) = # uncovered t-subsets ts with x in ts and ts\{x} ⊆ P.
        # ts\{x} ⊆ P means all t-1 other elements of ts are in P.
        # Equivalently: ts ∩ (V\P) ⊆ {x}, i.e. |ts ∩ (V\P)| <= 1 and x in ts.

        # Precompute for efficiency: for current P, compute compl_P = V\P.
        # For each x not in P: gain(x) = # uncovered ts with (ts_mask & compl_P_mask) ⊆ {x} and x in ts.
        # (ts_mask & compl_P_mask) is a bitmask of ts elements outside P.
        # We need that bitmask to be exactly {x} (a single bit) OR empty... but empty means ts ⊆ P already.
        # Since we only count ts NOT yet covered (ts ⊄ P), and ts ⊆ P∪{x} requires ts\{x} ⊆ P.
        # ts\{x} ⊆ P iff ts ∩ (V\P) ⊆ {x}. Since ts ⊄ P, ts ∩ (V\P) != empty, so ts ∩ (V\P) = {x}.
        # So gain(x) = # uncovered ts with (ts_mask & compl_P_mask) == (1<<x).

        while len(block) < k and time.time() < deadline - 2.0:
            compl_P = set(range(v)) - block_set
            compl_P_mask = 0
            for x in compl_P:
                compl_P_mask |= (1 << x)

            # For each uncovered t-subset, compute its intersection with compl_P
            inter = tsub_mask & compl_P_mask
            # For each x not in block, gain = count of uncovered ts with inter == (1<<x)
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                target = 1 << x
                # count uncovered ts where inter == target
                mask = (inter == target) & uncovered
                gain = int(np.sum(mask))
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == 0:
                        # no positive gain; break early to pick random
                        break
            if best_x == -1 or best_gain <= 0:
                # pick random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = int(rng.choice(candidates))
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        # Check duplicate
        if block in blocks:
            continue
        blocks.append(block)

        # Update uncovered: mark all t-subsets ⊆ block as covered
        block_mask = 0
        for x in block:
            block_mask |= (1 << x)
        compl_block_mask = 0
        for x in range(v):
            if x not in block_set:
                compl_block_mask |= (1 << x)
        inter = tsub_mask & compl_block_mask
        newly = (inter == 0) & uncovered
        uncovered[~newly] &= ~newly  # uncovered[newly] = False
        uncovered[newly] = False

    return blocks


def _build_cover_index(v, k, t, blocks, all_tsubs, tsub_mask):
    """
    For each t-subset, list of blocks that cover it.
    Block covers ts iff ts ⊆ block iff (ts_mask & compl_block_mask) == 0.
    """
    n_tsubs = len(all_tsubs)
    cover_list = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(blocks):
        compl = set(range(v)) - set(block)
        compl_mask = 0
        for x in compl:
            compl_mask |= (1 << x)
        inter = tsub_mask & compl_mask
        covered_idx = np.where(inter == 0)[0]
        for i in covered_idx:
            cover_list[int(i)].append(bi)
    return cover_list


def _local_search(v, k, t, blocks, deadline, rng, all_tsubs, tsub_mask):
    """
    Local search: iteratively remove redundant blocks and try to replace blocks.
    """
    n_tsubs = len(all_tsubs)
    blocks = [list(b) for b in blocks]

    # Build cover count and cover list
    def rebuild():
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            compl = set(range(v)) - set(block)
            compl_mask = 0
            for x in compl:
                compl_mask |= (1 << x)
            inter = tsub_mask & compl_mask
            covered = np.where(inter == 0)[0]
            cover_count[covered] += 1
        return cover_count

    cover_count = rebuild()

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        order = list(range(len(blocks)))
        rng.shuffle(order)
        for bi in order:
            if bi >= len(blocks):
                continue
            if time.time() > deadline - 5.0:
                break
            # Check redundancy: all t-subsets covered by block bi are also covered by others.
            # i.e. for every ts with ts ⊆ blocks[bi], cover_count[ts] >= 2.
            block = blocks[bi]
            compl = set(range(v)) - set(block)
            compl_mask = 0
            for x in compl:
                compl_mask |= (1 << x)
            inter = tsub_mask & compl_mask
            covered = np.where(inter == 0)[0]
            if len(covered) == 0:
                # block covers nothing? remove it
                del blocks[bi]
                cover_count = rebuild()
                improved = True
                continue
            # Check if any covered t-subset has cover_count == 1 (uniquely covered)
            if np.all(cover_count[covered] >= 2):
                # redundant
                del blocks[bi]
                cover_count = rebuild()
                improved = True
                continue
            # Try replacement: find t-subsets uniquely covered by this block
            unique_mask = (cover_count == 1) & (inter == 0)
            unique_idx = np.where(unique_mask)[0]
            if len(unique_idx) == 0:
                continue
            # The replacement block must contain all points in the union of unique t-subsets
            required = set()
            for i in unique_idx:
                required.update(all_tsubs[int(i)])
            if len(required) > k:
                continue
            need_more = k - len(required)
            remaining = [x for x in range(v) if x not in required]
            # Try a few random completions
            found = False
            for _ in range(5):
                if need_more == 0:
                    new_block = sorted(required)
                else:
                    extra = rng.sample(remaining, need_more)
                    new_block = sorted(required | set(extra))
                # Check: does new_block cover all unique t-subsets? (it should, since required ⊆ new_block)
                # But also must not lose coverage: new_block must cover all t-subsets that block covered uniquely.
                # Since required ⊆ new_block, all unique t-subsets (which are subsets of required) are covered.
                # Replace
                new_compl = set(range(v)) - set(new_block)
                new_compl_mask = 0
                for x in new_compl:
                    new_compl_mask |= (1 << x)
                new_inter = tsub_mask & new_compl_mask
                new_covered = np.where(new_inter == 0)[0]
                # Verify all unique are covered
                if not np.all(np.isin(unique_idx, new_covered)):
                    continue
                # Also verify new block doesn't break: it's fine as long as unique still covered.
                blocks[bi] = new_block
                cover_count = rebuild()
                improved = True
                found = True
                break

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    return unique_blocks


def construct(v, k, t):
    deadline = time.time() + 118.0  # safety margin before 120s kill
    rng = random.Random(42)

    all_tsubs, tsub_mask, point_mask = _build_masks(v, k, t)

    # Phase 1: Fast greedy baseline
    blocks = _fast_greedy(v, k, t, rng, deadline, all_tsubs, tsub_mask, point_mask)
    best = blocks
    best_len = len(blocks)

    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        refined = _local_search(v, k, t, blocks, deadline, rng, all_tsubs, tsub_mask)
        if len(refined) < best_len:
            best = refined
            best_len = len(refined)

    # Phase 3: Multiple greedy restarts with different seeds
    seed = 101
    while time.time() < deadline - 20.0:
        rng2 = random.Random(seed)
        candidate = _fast_greedy(v, k, t, rng2, deadline - 8.0, all_tsubs, tsub_mask, point_mask)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            if time.time() < deadline - 10.0:
                refined = _local_search(v, k, t, candidate, deadline, rng2, all_tsubs, tsub_mask)
                if len(refined) < best_len:
                    best = refined
                    best_len = len(refined)
        seed += 7

    return best
