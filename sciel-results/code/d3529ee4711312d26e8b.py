"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Covering design C(v, k, t) using a multi-phase approach:
    1. Greedy construction with multiple anchor strategies.
    2. Aggressive redundancy removal.
    3. Local search: block replacement to enable further reductions.
    4. Multiple restarts with different random seeds to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets
    tsets = list(itertools.combinations(range(v), t))
    N = len(tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(tsets)}

    # Precompute for each t-subset, the list of points it contains
    # and for each point, the list of t-subsets it belongs to
    point_tsets = [[] for _ in range(v)]
    for i, combo in enumerate(tsets):
        for x in combo:
            point_tsets[x].append(i)

    def greedy_construct(seed):
        """Greedy covering with randomized tie-breaking."""
        random.seed(seed)
        coverage = [0] * N
        blocks = []

        # Track uncovered count for fast termination
        uncovered_count = N

        while uncovered_count > 0:
            # Find an uncovered t-subset
            # For efficiency, pick a random uncovered one
            # We could maintain a list of uncovered, but let's just scan with random start
            found = False
            start = random.randint(0, N - 1)
            for offset in range(N):
                i = (start + offset) % N
                if coverage[i] == 0:
                    anchor = tsets[i]
                    found = True
                    break
            if not found:
                break

            # Build block greedily
            block_set = set(anchor)
            block_list = list(anchor)

            # Compute current gain for each point not in block
            # gain[x] = number of uncovered t-subsets containing x
            # We'll compute this incrementally
            gain = [0] * v
            for x in range(v):
                if x in block_set:
                    continue
                g = 0
                for ti in point_tsets[x]:
                    if coverage[ti] == 0:
                        g += 1
                gain[x] = g

            while len(block_list) < k:
                # Pick point with max gain, random tie-break
                best_g = -1
                candidates = []
                for x in range(v):
                    if x not in block_set:
                        if gain[x] > best_g:
                            best_g = gain[x]
                            candidates = [x]
                        elif gain[x] == best_g:
                            candidates.append(x)
                if not candidates:
                    break
                best_x = random.choice(candidates)
                block_set.add(best_x)
                block_list.append(best_x)

                # Update gain: for each uncovered t-subset containing best_x,
                # decrement gain for all other points in that t-subset
                for ti in point_tsets[best_x]:
                    if coverage[ti] == 0:
                        for y in tsets[ti]:
                            if y != best_x and y not in block_set:
                                gain[y] -= 1

            block_list.sort()
            blocks.append(block_list)

            # Update coverage
            for tcombo in itertools.combinations(block_list, t):
                i = tset_to_idx[tcombo]
                if coverage[i] == 0:
                    uncovered_count -= 1
                coverage[i] += 1

        return blocks, coverage

    def remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t):
        """Iteratively remove redundant blocks."""
        # Precompute t-subset indices for each block
        block_tsets = []
        for bl in blocks:
            ts = [tset_to_idx[tc] for tc in itertools.combinations(bl, t)]
            block_tsets.append(ts)

        changed = True
        while changed:
            changed = False
            for bi in range(len(blocks) - 1, -1, -1):
                redundant = True
                for i in block_tsets[bi]:
                    if coverage[i] <= 1:
                        redundant = False
                        break
                if redundant:
                    for i in block_tsets[bi]:
                        coverage[i] -= 1
                    blocks.pop(bi)
                    block_tsets.pop(bi)
                    changed = True

        return blocks, coverage, block_tsets

    def local_search(blocks, coverage, block_tsets, tsets, tset_to_idx, v, k, t, max_steps=20000):
        """Local search: try to replace blocks to enable reductions."""
        steps = 0
        improved = True
        while improved and steps < max_steps:
            improved = False
            steps += 1

            for bi in range(len(blocks)):
                # Calculate holes if we remove block bi
                holes = []
                for i in block_tsets[bi]:
                    if coverage[i] == 1:
                        holes.append(i)

                if not holes:
                    # Block is redundant, remove it
                    for i in block_tsets[bi]:
                        coverage[i] -= 1
                    blocks.pop(bi)
                    block_tsets.pop(bi)
                    improved = True
                    break

                # We need to cover these holes with a new block
                U = set()
                for h_idx in holes:
                    U.update(tsets[h_idx])

                if len(U) > k:
                    continue

                # Find a good replacement block
                # The new block must contain U
                remaining_points = [x for x in range(v) if x not in U]
                need = k - len(U)
                if need < 0 or need > len(remaining_points):
                    continue

                # Score potential extra points by how many additional t-subsets they cover
                # that are currently covered by only 1 block (to enable further removals)
                # Simplified: just pick random extras, but try a few candidates
                best_new_block = None
                best_score = -1

                for attempt in range(5):
                    if need == 0:
                        extra = []
                    else:
                        extra = random.sample(remaining_points, need)
                    new_block_set = U.union(extra)
                    new_block = sorted(new_block_set)

                    # Check if already in blocks
                    if new_block in blocks:
                        continue

                    # Compute score: number of t-subsets in new_block that are
                    # currently covered by exactly 1 block (excluding bi)
                    score = 0
                    for tc in itertools.combinations(new_block, t):
                        i = tset_to_idx[tc]
                        if i in [h for h in holes]:
                            continue  # already counted
                        if 1 < coverage[i] <= 2:
                            score += 1

                    if score > best_score:
                        best_score = score
                        best_new_block = new_block

                if best_new_block is None:
                    continue

                # Perform the swap
                for i in block_tsets[bi]:
                    coverage[i] -= 1

                new_block_tsets = [tset_to_idx[tc] for tc in itertools.combinations(best_new_block, t)]
                for i in new_block_tsets:
                    coverage[i] += 1

                blocks[bi] = best_new_block
                block_tsets[bi] = new_block_tsets

                # Check for new redundancies
                for ri in range(len(blocks) - 1, -1, -1):
                    if ri == bi:
                        continue
                    redundant = True
                    for i in block_tsets[ri]:
                        if coverage[i] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for i in block_tsets[ri]:
                            coverage[i] -= 1
                        blocks.pop(ri)
                        block_tsets.pop(ri)
                        improved = True
                        break

                if improved:
                    break

        return blocks, coverage, block_tsets

    # Main: try multiple restarts
    best_blocks = None
    best_count = float('inf')

    num_restarts = 10  # Will be limited by time

    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break

        seed = 42 + restart * 137
        blocks, coverage = greedy_construct(seed)

        # Remove redundant
        blocks, coverage, block_tsets = remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t)

        # Local search
        if time.time() < deadline - 15.0:
            blocks, coverage, block_tsets = local_search(blocks, coverage, block_tsets, tsets, tset_to_idx, v, k, t, max_steps=10000)

        # Final redundancy pass
        blocks, coverage, block_tsets = remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

        # If we've hit a good count, we can reduce remaining restarts
        if best_count <= 300:
            break

    if best_blocks is None:
        # Fallback: simple greedy
        blocks, _ = greedy_construct(0)
        return blocks

    return best_blocks

# EVOLVE-BLOCK-END
