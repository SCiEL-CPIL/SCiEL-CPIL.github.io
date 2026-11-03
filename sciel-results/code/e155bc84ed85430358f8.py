"""Combined LP-guided greedy with multi-restart and deletion for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Precompute: for each k-subset, which t-subsets it contains
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    # Solve LP to get dual weights
    duals = None
    try:
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
    except Exception:
        duals = None

    # Fallback duals if LP fails: use uniform or degree-based
    if duals is None or not np.any(duals > 0):
        # Use degree-based heuristic: subsets covered by fewer blocks are more valuable
        duals = np.zeros(n_t)
        for i in range(n_t):
            deg = len(blocks_of_subset[i])
            duals[i] = 1.0 / max(1, deg)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 8  # Fewer restarts since each is more expensive with LP guidance

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        # Dynamic LP-guided greedy with perturbation for diversity
        covered = np.zeros(n_t, dtype=bool)
        covered_count = 0
        blocks = []
        picked = np.zeros(n_k, dtype=bool)

        # Initialize block weights from duals
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])

        # Add perturbation for diversity across restarts
        if restart > 0:
            noise_scale = 0.1 * restart / n_restarts
            block_weights = block_weights + random.random() * noise_scale

        # Heap for dynamic greedy
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)

        while not np.all(covered) and time_ok():
            neg_w, j = heapq.heappop(heap)
            w = -neg_w

            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                continue

            # Select block j
            picked[j] = True
            blocks.append(list(k_subsets[j]))

            # Update: mark covered subsets and update weights
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    covered_count += 1
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))

        # Check completeness
        if not np.all(covered):
            # Fallback: greedy for remaining uncovered
            remaining = [i for i in range(n_t) if not covered[i]]
            for _ in range(100):  # limited fallback iterations
                if not remaining or not time_ok():
                    break
                best_j = -1
                best_gain = 0
                for j in range(n_k):
                    if picked[j]:
                        continue
                    gain = sum(1 for s in subsets_of_block[j] if not covered[s])
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j < 0 or best_gain == 0:
                    break
                picked[best_j] = True
                blocks.append(list(k_subsets[best_j]))
                for s_idx in subsets_of_block[best_j]:
                    covered[s_idx] = True
                remaining = [i for i in range(n_t) if not covered[i]]

        # Deletion phase: remove redundant blocks
        if time_ok() and np.all(covered):
            tsub_block_count = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                for s in itertools.combinations(b, t):
                    tsub_block_count[t_index[s]] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for s in itertools.combinations(blocks[i], t):
                        if tsub_block_count[t_index[s]] <= 1:
                            removable = False
                            break
                    if removable:
                        for s in itertools.combinations(blocks[i], t):
                            tsub_block_count[t_index[s]] -= 1
                        blocks.pop(i)
                        changed = True
                        break

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        # Early stop if very good
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
