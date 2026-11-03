import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # 1. Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    # 2. Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    # 3. Build sparse incidence matrix A: A[i, j] = 1 if t_subs[j] ⊆ k_subs[i]
    row_indices = []
    col_indices = []
    data = []
    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    # Precompute column slices for each block for fast updates
    block_cols = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n_k)]

    def greedy_cover(weights=None, seed_blocks=None):
        """Greedy set cover with optional frequency-based weighting."""
        u = np.ones(n_t, dtype=np.float64)
        selected_blocks = []

        if seed_blocks:
            for idx in seed_blocks:
                u[block_cols[idx]] = 0
                selected_blocks.append(idx)

        if weights is None:
            weights = np.ones(n_t, dtype=np.float64)

        while u.sum() > 0:
            if time.time() > deadline:
                break

            # Compute weighted gain: sum of weights of uncovered t-subsets in each block
            gains = (A * weights) @ u  # element-wise multiply then matrix-vector product
            
            # Actually, A @ (weights * u) is more efficient
            gains = A @ (weights * u)

            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]

            if best_gain <= 0:
                break

            selected_blocks.append(best_idx)
            u[block_cols[best_idx]] = 0

            # Update weights: increase weight of still-uncovered t-subsets
            # to prioritize scarce coverage
            still_uncovered = u > 0
            if still_uncovered.sum() > 0:
                weights[still_uncovered] *= 1.1

        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering."""
        if not selected_blocks:
            return selected_blocks

        selected_t_subs = [block_cols[idx] for idx in selected_blocks]

        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            cover_count[cols] += 1

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                # Check if any col in cols has cover_count == 1
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break

                if not is_essential:
                    to_remove.append(i)

            if to_remove:
                changed = True
                for i in sorted(to_remove, reverse=True):
                    cols = selected_t_subs[i]
                    cover_count[cols] -= 1
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)

        return selected_blocks

    def local_improve(selected_blocks, max_rounds=5):
        """Local search: try swapping blocks to reduce total count."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        best_t_subs = [block_cols[idx] for idx in best]

        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False

            # Build coverage counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for cols in best_t_subs:
                cover_count[cols] += 1

            # Try removing each block and see if we can replace it with fewer blocks
            for i in range(len(best)):
                if time.time() > deadline:
                    break

                # Temporarily remove block i
                cols_to_remove = best_t_subs[i]
                cover_count[cols_to_remove] -= 1

                # Find which t-subsets become uncovered (cover_count == 0)
                uncovered_after_removal = set()
                for c in cols_to_remove:
                    if cover_count[c] == 0:
                        uncovered_after_removal.add(c)

                if not uncovered_after_removal:
                    # Block i is redundant, remove it
                    best.pop(i)
                    best_t_subs.pop(i)
                    improved = True
                    break

                # Try to find a single block that covers all uncovered t-subsets
                # and doesn't cause issues
                found_replacement = False
                for candidate in range(n_k):
                    if candidate in best:
                        continue
                    cand_cols = set(block_cols[candidate])
                    if uncovered_after_removal.issubset(cand_cols):
                        # This block covers all the uncovered t-subsets
                        # Add it and remove block i
                        best[i] = candidate
                        best_t_subs[i] = block_cols[candidate]
                        cover_count[block_cols[candidate]] += 1
                        found_replacement = True
                        improved = True
                        break

                if not found_replacement:
                    # Restore cover_count
                    cover_count[cols_to_remove] += 1

                if improved:
                    break

            if not improved:
                break

        return best

    # Run multiple randomized restarts
    best_result = None
    best_size = float('inf')

    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            # First restart: standard greedy
            result = greedy_cover()
        else:
            # Subsequent restarts: use frequency-weighted greedy with random initialization
            random.seed(restart * 12345)
            # Start with a few random blocks to break symmetry
            seed_blocks = []
            n_seed = min(5, n_k)
            for _ in range(n_seed):
                idx = random.randint(0, n_k - 1)
                if idx not in seed_blocks:
                    seed_blocks.append(idx)
            result = greedy_cover(seed_blocks=seed_blocks)

        # Post-process: remove redundant blocks
        result = remove_redundant(result)

        # Local improvement
        result = local_improve(result, max_rounds=3)

        if len(result) < best_size:
            best_size = len(result)
            best_result = result

    if best_result is None:
        # Fallback: simple greedy
        best_result = greedy_cover()
        best_result = remove_redundant(best_result)

    return [list(k_subs[idx]) for idx in best_result]
