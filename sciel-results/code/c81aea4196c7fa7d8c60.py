import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers
    # Using a flat array for faster NumPy operations
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    best_solution = None
    best_score = float('inf')

    def aggressive_deletion(current_blocks, coverage_count):
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 4.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def rebuild_coverage(block_list):
        cc = np.zeros(n_ts, dtype=np.int32)
        for b in block_list:
            cc[block_t_indices[b]] += 1
        return cc

    def try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
        """Try replacing old_x with new_x in block b_idx. Returns True if valid."""
        if new_x == old_x:
            return False
        block = all_ks[b_idx]
        if new_x in block:
            return False

        other = [p for p in block if p != old_x]
        lost = []
        gained = []
        for rest in itertools.combinations(other, t - 1):
            lost.append(idx_map[tuple(sorted(rest + (old_x,)))])
            gained.append(idx_map[tuple(sorted(rest + (new_x,)))])
        lost = np.array(lost, dtype=np.int32)
        gained = np.array(gained, dtype=np.int32)

        # After swap, coverage for 'lost' drops by 1; must stay >= 1.
        if np.any(coverage_count[lost] <= 1):
            return False

        coverage_count[lost] -= 1
        coverage_count[gained] += 1
        return True

    def local_search(blocks, coverage_count, max_iters):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False

            # Pass 1: try to delete any redundant blocks
            for b_idx in blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    coverage_count[subs_in_block] -= 1
                    blocks.remove(b_idx)
                    improved = True

            # Pass 2: point-swap perturbations
            if len(blocks) == 0:
                break
            for b_idx in blocks:
                if time.time() - start_time > time_limit - 5.0:
                    break
                block = all_ks[b_idx]
                # Try swapping each point in the block with a random outside point
                for old_x in block:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    for new_x in candidates[:5]:
                        if try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
                            improved = True
                            break
                    if improved:
                        break

            # After any swap, try deleting again
            if improved:
                for b_idx in list(blocks):
                    subs_in_block = block_t_indices[b_idx]
                    if np.all(coverage_count[subs_in_block] > 1):
                        coverage_count[subs_in_block] -= 1
                        blocks.remove(b_idx)

            if not improved:
                break

        return blocks

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)

        cover_count = np.zeros(n_ts, dtype=np.int32)
        blocks_indices = []
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 6.0:
                return None, None

            # Calculate Gains and Penalties
            # Gains: number of triples in block that are currently uncovered
            uncovered_mask = (cover_count == 0)
            newly_covered = uncovered_mask[block_t_indices]
            gains = newly_covered.sum(axis=1)
            
            # Filter blocks that add no new coverage
            valid_mask = gains > 0
            if not np.any(valid_mask):
                break

            # Penalties: sum of current coverage counts for all triples in the block
            # This penalizes blocks that overlap heavily with existing blocks
            penalties = cover_count[block_t_indices].sum(axis=1)
            
            # Score: Ratio of new coverage to redundancy
            # Add 1 to denominator to avoid division by zero and to provide a baseline
            scores = np.zeros(n_ks, dtype=np.float32)
            scores[valid_mask] = gains[valid_mask].astype(np.float32) / (1.0 + penalties[valid_mask].astype(np.float32))
            
            # Find the best block
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            # Get candidates with high scores (top 10% or top K)
            # To add randomness, we pick from the top candidates
            threshold = max_score * 0.95  # Consider blocks with score >= 95% of max
            candidates = np.where(scores >= threshold)[0]
            
            if len(candidates) > 10:
                # Randomly sample from top candidates
                chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                chosen_idx = candidates[random.randint(0, len(candidates) - 1)]

            blocks_indices.append(chosen_idx)

            # Update cover_count
            subs_to_cover = block_t_indices[chosen_idx]
            newly_covered_count = np.sum(uncovered_mask[subs_to_cover])
            remaining -= newly_covered_count
            cover_count[subs_to_cover] += 1

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)

        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Local search with point swaps
        if time.time() - start_time < time_limit - 10.0:
            current_blocks = local_search(current_blocks, coverage_count, max_iters=100)
            coverage_count = rebuild_coverage(current_blocks)

        # Final deletion pass
        current_blocks = aggressive_deletion(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    # Phase 1: Multiple restarts with the new ratio-based greedy
    restart_count = 0
    while time.time() - start_time < time_limit - 20.0:
        seed = restart_count
        blocks, score = run_greedy(seed)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    # Phase 2: Final high-precision polish on the best solution found
    if best_solution is not None and time.time() - start_time < time_limit - 8.0:
        coverage_count = rebuild_coverage(best_solution)
        # Tighter deletion
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

        # Final point-swap polish
        if time.time() - start_time < time_limit - 5.0:
            best_solution = local_search(best_solution, coverage_count, max_iters=200)
            coverage_count = rebuild_coverage(best_solution)
            best_solution = aggressive_deletion(best_solution, coverage_count)

    if best_solution is None:
        # Fallback: simple greedy without ratio, just in case
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
