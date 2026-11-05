# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Fixed deadline guard
_DEADLINE = 145.0  # seconds, leaving 5s margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def construct(v, k, t):
    start_time = time.time()
    random.seed(42)

    # Precompute all t-subsets and assign integer indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets and the t-subsets they contain
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, store the list of t-subset indices it covers
    ksub_to_tsubs = []
    for ks in all_ksubs:
        covered = [tsub_index[ts] for ts in itertools.combinations(ks, t)]
        ksub_to_tsubs.append(covered)

    # Convert to numpy arrays for fast operations
    # ksub_to_tsubs is a list of lists; pad to a 2D array
    max_tsubs_per_ksub = len(all_tsubs)  # worst case, but actually C(k,t)
    # Actually each k-subset covers exactly C(k,t) t-subsets
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build a 2D numpy array: shape (n_ksubs, c_k_t)
    ksub_tsub_matrix = np.zeros((n_ksubs, c_k_t), dtype=np.int32)
    for i, covered in enumerate(ksub_to_tsubs):
        ksub_tsub_matrix[i, :len(covered)] = covered

    # Phase 1: Greedy covering to get an initial valid covering
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    selected_blocks = []  # list of k-subset indices

    # Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets
    uncovered_count = np.ones(n_tsubs, dtype=np.int32)  # 1 if uncovered, 0 if covered

    # Precompute for each k-subset which t-subsets it covers (already done)
    # Greedy loop
    while np.any(uncovered_count > 0):
        if time.time() - start_time > _DEADLINE:
            break

        # For each candidate k-subset, compute how many uncovered t-subsets it covers
        # Only consider k-subsets not already selected
        best_gain = -1
        best_ksub = -1

        # Sample candidates to speed up
        n_candidates = min(n_ksubs, 5000)
        candidates = random.sample(range(n_ksubs), n_candidates) if n_ksubs > n_candidates else list(range(n_ksubs))

        for ki in candidates:
            if ki in selected_blocks:
                continue
            tsubs = ksub_tsub_matrix[ki]
            gain = np.sum(uncovered_count[tsubs])
            if gain > best_gain:
                best_gain = gain
                best_ksub = ki

        if best_ksub == -1:
            # Fallback: try all
            for ki in range(n_ksubs):
                if ki in selected_blocks:
                    continue
                tsubs = ksub_tsub_matrix[ki]
                gain = np.sum(uncovered_count[tsubs])
                if gain > best_gain:
                    best_gain = gain
                    best_ksub = ki

        if best_ksub == -1 or best_gain <= 0:
            break

        selected_blocks.append(best_ksub)
        tsubs = ksub_tsub_matrix[best_ksub]
        coverage[tsubs] += 1
        uncovered_count[tsubs] = np.minimum(uncovered_count[tsubs], 0)

    if not selected_blocks:
        # Fallback: brute force add all k-subsets (shouldn't happen)
        selected_blocks = list(range(n_ksubs))
        for ki in selected_blocks:
            tsubs = ksub_tsub_matrix[ki]
            coverage[tsubs] += 1

    # Phase 2: Remove redundant blocks
    # A block is redundant if removing it doesn't drop any t-subset coverage below 1
    # Iterate until no more removals
    changed = True
    while changed:
        if time.time() - start_time > _DEADLINE:
            break
        changed = False
        # Shuffle order for diversity
        order = list(range(len(selected_blocks)))
        random.shuffle(order)
        for idx in order:
            ki = selected_blocks[idx]
            tsubs = ksub_tsub_matrix[ki]
            # Check if all t-subsets in this block have coverage >= 2
            if np.all(coverage[tsubs] >= 2):
                coverage[tsubs] -= 1
                selected_blocks.pop(idx)
                changed = True
                break  # Restart after removal

    # Phase 3: Dual local search - swap redundant blocks with better candidates
    # Try to swap a block with another that maintains coverage while allowing more removals
    for iteration in range(50):
        if time.time() - start_time > _DEADLINE:
            break
        progress = False

        # Try to remove blocks first
        for idx in range(len(selected_blocks)):
            ki = selected_blocks[idx]
            tsubs = ksub_tsub_matrix[ki]
            if np.all(coverage[tsubs] >= 2):
                coverage[tsubs] -= 1
                selected_blocks.pop(idx)
                progress = True
                break

        if progress:
            continue

        # Try swaps: replace block i with block j if it allows subsequent removal
        # For efficiency, try a few random swaps
        n_tries = min(100, len(selected_blocks) * 10)
        for _ in range(n_tries):
            if time.time() - start_time > _DEADLINE:
                break
            i = random.randint(0, len(selected_blocks) - 1)
            j = random.randint(0, n_ksubs - 1)
            if j in selected_blocks:
                continue

            ki = selected_blocks[i]
            kj = j

            tsubs_i = ksub_tsub_matrix[ki]
            tsubs_j = ksub_tsub_matrix[kj]

            # Simulate: remove ki, add kj
            # Check if after removal, all tsubs_i that aren't in tsubs_j still have coverage >= 1
            # And after adding kj, no t-subset exceeds... (no upper bound, so just check removal safety)

            # tsubs_i that are NOT covered by kj need coverage >= 2 (since we're removing ki)
            # tsubs_j that are NOT covered by ki just get +1 (always safe)

            # Efficient check: for tsubs in tsubs_i, if coverage[tsub] == 1 and tsub not in tsubs_j, then unsafe
            # We need: for all tsub in tsubs_i, coverage[tsub] >= 2 OR tsub in tsubs_j

            # Build set of tsubs_j for fast lookup
            tsubs_j_set = set(tsubs_j.tolist())

            safe = True
            for ts in tsubs_i:
                if coverage[ts] == 1 and ts not in tsubs_j_set:
                    safe = False
                    break

            if safe:
                # Perform swap
                coverage[tsubs_i] -= 1
                coverage[tsubs_j] += 1
                selected_blocks[i] = kj
                progress = True

                # Try to remove more after swap
                for idx in range(len(selected_blocks)):
                    kx = selected_blocks[idx]
                    tsubs_x = ksub_tsub_matrix[kx]
                    if np.all(coverage[tsubs_x] >= 2):
                        coverage[tsubs_x] -= 1
                        selected_blocks.pop(idx)
                        break

        if not progress:
            break

    # Convert selected block indices back to lists of points
    result = [list(all_ksubs[ki]) for ki in selected_blocks]
    result.sort()
    return result
# EVOLVE-BLOCK-END
