import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)

    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_covered = []
    for b in all_blocks:
        covered = []
        for sub in itertools.combinations(b, t):
            covered.append(subset_to_idx[sub])
        block_covered.append(covered)

    # Greedy set cover: repeatedly pick block with max marginal gain
    def greedy_set_cover():
        uncovered = [True] * n_subsets
        uncovered_count = n_subsets
        selected = []
        remaining = set(range(n_blocks))

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0

            for bi in remaining:
                gain = 0
                for idx in block_covered[bi]:
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                elif gain == best_gain and gain > 0 and random.random() < 0.03:
                    best_block = bi

            if best_block == -1:
                break

            selected.append(best_block)
            remaining.discard(best_block)

            for idx in block_covered[best_block]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1

        return selected

    # Local search: try swapping one selected block with one unselected block
    def local_search(selected, max_iters=2000):
        if time.time() > deadline:
            return selected

        # Build coverage count for each t-subset
        cover_count = [0] * n_subsets
        selected_set = set(selected)
        for bi in selected:
            for idx in block_covered[bi]:
                cover_count[idx] += 1

        # Check validity
        for c in cover_count:
            if c < 1:
                return selected

        current_size = len(selected)
        improved = True
        iter_count = 0

        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1

            # Try removing each selected block
            for i, bi in enumerate(selected):
                # Check if bi is removable (all its covered subsets have count > 1)
                removable = True
                for idx in block_covered[bi]:
                    if cover_count[idx] == 1:
                        removable = False
                        break
                if not removable:
                    continue

                # Try removing bi
                for idx in block_covered[bi]:
                    cover_count[idx] -= 1
                new_selected = selected[:i] + selected[i+1:]
                new_set = selected_set - {bi}

                # Try adding any unselected block that covers a subset now at count 1
                # Actually, we just need to check if removing is safe (all counts >= 1)
                all_covered = True
                for c in cover_count:
                    if c < 1:
                        all_covered = False
                        break

                if all_covered:
                    # Safe removal!
                    selected = new_selected
                    selected_set = new_set
                    improved = True
                    break
                else:
                    # Restore
                    for idx in block_covered[bi]:
                        cover_count[idx] += 1

        return selected

    # Try a few greedy runs with different random seeds, then local search
    best_blocks = None
    best_size = float('inf')

    n_attempts = 0
    while time.time() < deadline:
        n_attempts += 1
        random.seed(n_attempts * 7919)

        selected = greedy_set_cover()
        if time.time() > deadline:
            break

        # Local search to reduce
        selected = local_search(selected)

        if len(selected) < best_size:
            best_size = len(selected)
            best_blocks = list(selected)

        # If we've gotten a good result, do more local search
        if best_size <= 160 and time.time() < deadline - 5:
            selected = local_search(selected, max_iters=5000)
            if len(selected) < best_size:
                best_size = len(selected)
                best_blocks = list(selected)

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = [True] * n_subsets
        uncovered_count = n_subsets
        selected = []
        remaining = set(range(n_blocks))
        while uncovered_count > 0:
            best_block = -1
            best_gain = 0
            for bi in remaining:
                gain = sum(1 for idx in block_covered[bi] if uncovered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            if best_block == -1:
                break
            selected.append(best_block)
            remaining.discard(best_block)
            for idx in block_covered[best_block]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        best_blocks = selected

    return [list(all_blocks[bi]) for bi in best_blocks]
