import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    N = 3 ** n  # 729
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute neighbor lists (sphere of radius 1)
    # neighbor_list[i] = list of indices covered by words[i] (including i itself)
    neighbor_list = []
    for i in range(N):
        w_i = words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                neighbors.append(j)
        neighbor_list.append(neighbors)

    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbor_list]

    def is_valid_code(code_set):
        """Check if a set of codeword indices covers all words."""
        if not code_set:
            return False
        coverage = np.zeros(N, dtype=np.int32)
        for idx in code_set:
            coverage[neighbor_arrays[idx]] += 1
        return not np.any(coverage == 0)

    def remove_redundant(code_set, coverage_counts):
        """Remove redundant codewords. Returns (new_code_set, new_coverage_counts, removed_count)."""
        changed = True
        removed = 0
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if idx is redundant: all its neighbors have coverage > 1
                redundant = True
                for j in neighbor_arrays[idx]:
                    if coverage_counts[j] == 1:
                        redundant = False
                        break
                if redundant:
                    coverage_counts[neighbor_arrays[idx]] -= 1
                    code_set.remove(idx)
                    changed = True
                    removed += 1
        return code_set, coverage_counts, removed

    def greedy_set_cover(random_seed=None):
        """Greedy set cover to find a covering code."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random

        uncovered = np.ones(N, dtype=bool)
        code_set = set()
        coverage_counts = np.zeros(N, dtype=np.int32)

        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break

            best_idx = -1
            best_count = 0

            # For speed, we can sample candidates or do full search
            # Full search over 729 candidates, each checking up to 13 neighbors
            # This is 729 * 13 = ~9500 operations per iteration, very fast
            for i in range(N):
                if i in code_set:
                    continue
                # Count how many currently uncovered words are covered by i
                nbrs = neighbor_arrays[i]
                count = np.sum(uncovered[nbrs])
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if best_count == np.sum(uncovered):
                        break  # Can't do better than covering all remaining

            if best_idx == -1 or best_count == 0:
                break

            code_set.add(best_idx)
            nbrs = neighbor_arrays[best_idx]
            uncovered[nbrs] = False
            coverage_counts[nbrs] += 1

        # Remove redundant codewords
        code_set, coverage_counts, _ = remove_redundant(code_set, coverage_counts)

        # Verify validity
        if not is_valid_code(code_set):
            return None, None

        return code_set, coverage_counts

    # Phase 1: Multiple greedy restarts with different random seeds
    best_code = None
    best_size = float('inf')
    best_coverage = None

    # First, try a deterministic greedy (no randomization)
    code, cov = greedy_set_cover(random_seed=0)
    if code is not None and len(code) < best_size:
        best_size = len(code)
        best_code = set(code)
        best_coverage = cov.copy()

    # Try several more restarts with different seeds to find diverse solutions
    num_restarts = 30
    for seed in range(1, num_restarts + 1):
        if time.time() > deadline - 40.0:
            break
        code, cov = greedy_set_cover(random_seed=seed * 12345)
        if code is not None and len(code) < best_size:
            best_size = len(code)
            best_code = set(code)
            best_coverage = cov.copy()

    # Phase 2: Local search refinement on the best solution
    if best_code is not None:
        current_code = set(best_code)
        current_coverage = best_coverage.copy()

        # Try to improve via swap operations
        T = 5.0
        T_min = 0.1
        cooling = 0.9995

        all_indices = list(range(N))

        # Precompute for faster redundant checks
        def can_remove(idx, cov):
            """Check if removing idx keeps coverage valid."""
            for j in neighbor_arrays[idx]:
                if cov[j] == 1:
                    return False
            return True

        def swap_improve(current_code, current_coverage):
            """Try swaps to improve the solution."""
            improved = False
            code_list = list(current_code)
            for to_remove in code_list:
                if time.time() > deadline - 15.0:
                    return current_code, current_coverage, improved
                # Try to find a replacement that allows removal of to_remove
                # and possibly enables further reductions
                for to_add in all_indices:
                    if to_add in current_code:
                        continue
                    # Simulate: remove to_remove, add to_add
                    temp_cov = current_coverage.copy()
                    temp_cov[neighbor_arrays[to_remove]] -= 1
                    if np.any(temp_cov < 0):
                        continue
                    temp_cov[neighbor_arrays[to_add]] += 1
                    if np.any(temp_cov == 0):
                        continue
                    # Valid swap. Now try to remove redundant words
                    temp_code = set(current_code)
                    temp_code.remove(to_remove)
                    temp_code.add(to_add)
                    temp_code, temp_cov, removed = remove_redundant(temp_code, temp_cov)
                    if len(temp_code) < len(current_code):
                        current_code = temp_code
                        current_coverage = temp_cov
                        improved = True
                        break
                if improved:
                    break
            return current_code, current_coverage, improved

        while time.time() < deadline - 10.0 and T > T_min:
            move_type = random.random()

            if move_type < 0.6:
                # Try to remove a codeword
                if not current_code:
                    break
                candidates = [idx for idx in current_code if can_remove(idx, current_coverage)]
                if not candidates:
                    # No obviously redundant codeword, try a random one
                    to_remove = random.choice(list(current_code))
                else:
                    to_remove = random.choice(candidates)

                # Check if removal keeps validity
                newly_uncovered = 0
                for j in neighbor_arrays[to_remove]:
                    if current_coverage[j] == 1:
                        newly_uncovered += 1

                delta_cost = -1 + 1000 * newly_uncovered
                if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                    current_coverage[neighbor_arrays[to_remove]] -= 1
                    current_code.remove(to_remove)
                    if newly_uncovered == 0 and len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                        best_coverage = current_coverage.copy()

            else:
                # Try swap operations
                current_code, current_coverage, improved = swap_improve(current_code, current_coverage)
                if improved and len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                    best_coverage = current_coverage.copy()

            T *= cooling

        # Final redundancy removal
        current_code, current_coverage, _ = remove_redundant(current_code, current_coverage)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)
            best_coverage = current_coverage.copy()

    # Phase 3: Final verification and return
    if best_code is None or not is_valid_code(best_code):
        # Fallback to direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[i]) for i in best_code]
