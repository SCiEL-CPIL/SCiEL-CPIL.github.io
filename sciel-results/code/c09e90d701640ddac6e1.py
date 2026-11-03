"""
Covering design C(v,k,t): Vectorized Greedy (heap) + Exact MIS Deletion + Local-Search Swap Polish.

Strategy:
1. Fast heap-based greedy (lazy gain updates) -> many diverse initial coverings.
2. Iterative exact deletion via Maximum Independent Set on the redundancy/conflict graph.
3. Local-search polish: remove redundant blocks, then 1-for-1 swaps where a candidate
   block covers every t-subset that a current block uniquely covers.
"""
import itertools
import random
import time
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            block_tsubs[b_idx].append(idx)
            tsub_blocks[idx].append(b_idx)

    full_tsub_mask = (1 << n_tsubsets) - 1 if n_tsubsets < 1024 else None

    def block_tsub_indices(block):
        if isinstance(block, int):
            return block_tsubs[block]
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices

    def greedy_cover(seed):
        """Heap-based greedy with lazy gain updates."""
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0

        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = len(block_tsubs[b_idx])

        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)

        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    break

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1

            new_covered = block_tsubs[best_block]
            uncovered_mask &= ~(1 << best_block) if False else uncovered_mask
            # recompute uncovered bits
            for ts_idx in new_covered:
                uncovered_mask &= ~(1 << ts_idx)

            for ts_idx in new_covered:
                for b_idx2 in tsub_blocks[ts_idx]:
                    if not (selected_mask & (1 << b_idx2)) and gains[b_idx2] > 0:
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))

        return selected

    def exact_deletion_pass(blocks, deadline):
        """Iteratively remove redundant blocks via exact MIS on conflict graph."""
        blocks_list = [b for b in blocks]
        total_removed = 0

        while time.time() < deadline - 12:
            n = len(blocks_list)
            if n == 0:
                break

            cov_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsubs_cur = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs_cur.append(indices)
                cov_count[indices] += 1

            critical = [False] * n
            for bi in range(n):
                for idx in block_tsubs_cur[bi]:
                    if cov_count[idx] == 1:
                        critical[bi] = True
                        break

            non_critical_idx = [i for i in range(n) if not critical[i]]
            if len(non_critical_idx) < 2:
                break

            m = len(non_critical_idx)
            unique_sets = []
            for bi in non_critical_idx:
                unique = set()
                for idx in block_tsubs_cur[bi]:
                    if cov_count[idx] == 1:
                        unique.add(idx)
                unique_sets.append(unique)

            adj = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    if not (unique_sets[i] & unique_sets[j]):
                        adj[i].add(j)
                        adj[j].add(i)

            mis_size, mis_nodes = solve_mis(adj, m, deadline - 18)
            if mis_size < 2:
                break

            remove_set = set(non_critical_idx[i] for i in mis_nodes)
            blocks_list = [b for i, b in enumerate(blocks_list) if i not in remove_set]
            total_removed += len(remove_set)

        return blocks_list, total_removed

    def solve_mis(adj, n, deadline):
        """Maximum Independent Set via branch-and-bound with greedy coloring bound."""
        best_size = [0]
        best_nodes = [[]]

        def greedy_coloring(nodes, adj):
            if not nodes:
                return 0, []
            deg = {v: len(adj[v]) for v in nodes}
            ordered = sorted(nodes, key=lambda v: -deg[v])
            color_of = {}
            color_classes = []
            for v in ordered:
                used = set()
                for u in adj[v]:
                    if u in color_of:
                        used.add(color_of[u])
                c = 0
                while c in used:
                    c += 1
                color_of[v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(v)
            return len(color_classes), color_classes

        def branch(nodes, current_selection):
            if time.time() > deadline:
                return
            if not nodes:
                if len(current_selection) > best_size[0]:
                    best_size[0] = len(current_selection)
                    best_nodes[0] = list(current_selection)
                return

            n_colors, _ = greedy_coloring(nodes, adj)
            if len(current_selection) + n_colors <= best_size[0]:
                return

            nodes_set = set(nodes)
            v = max(nodes, key=lambda x: len(adj[x] & nodes_set))

            branch([u for u in nodes if u != v and u not in adj[v]],
                   current_selection + [v])
            branch([u for u in nodes if u != v], current_selection)

        branch(list(range(n)), [])
        return best_size[0], best_nodes[0]

    def swap_polish(blocks, deadline):
        """Remove redundant blocks and try 1-for-1 swaps."""
        blocks_list = [list(b) for b in blocks]
        improved = True

        while improved and time.time() < deadline - 8:
            improved = False
            n_blocks = len(blocks_list)

            cov_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsubs_cur = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs_cur.append(indices)
                cov_count[indices] += 1

            # Phase 1: remove any block with no uniquely-covered t-subset
            removable = [bi for bi in range(n_blocks)
                         if all(cov_count[idx] > 1 for idx in block_tsubs_cur[bi])]
            if removable:
                rem_set = set(removable)
                for bi in removable:
                    for idx in block_tsubs_cur[bi]:
                        cov_count[idx] -= 1
                blocks_list = [b for i, b in enumerate(blocks_list) if i not in rem_set]
                improved = True
                continue

            # Phase 2: 1-for-1 swap
            current_blocks = list(range(n_blocks))
            rng = random.Random(1000 + len(blocks_list))
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > deadline - 8:
                    break
                unique_tsubs = [idx for idx in block_tsubs_cur[bi] if cov_count[idx] == 1]
                if not unique_tsubs:
                    for idx in block_tsubs_cur[bi]:
                        cov_count[idx] -= 1
                    blocks_list.pop(bi)
                    improved = True
                    break

                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                for idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[idx])
                candidates.discard(bi)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                if len(cand_list) > 30:
                    cand_list = cand_list[:30]

                swapped = False
                for r in cand_list:
                    if r in blocks_list:
                        continue
                    for idx in block_tsubs_cur[bi]:
                        cov_count[idx] -= 1
                    blocks_list[bi] = list(all_blocks[r])
                    for idx in block_tsub_indices(r):
                        cov_count[idx] += 1
                    improved = True
                    swapped = True
                    break
                if swapped:
                    break

        return blocks_list

    # Phase 1: many diverse greedy restarts
    best_blocks = None
    best_count = float('inf')

    seed = 0
    while time.time() < deadline - 40:
        blocks = greedy_cover(seed)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1

    # Phase 2: exact deletion on the best
    if best_blocks is not None and time.time() < deadline - 18:
        best_blocks, _ = exact_deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)

    # Phase 3: swap polish
    if time.time() < deadline - 8:
        best_blocks = swap_polish(best_blocks, deadline)
        best_count = len(best_blocks)

    # Phase 4: refine a couple of other competitive starts
    for start_seed in range(4):
        if time.time() > deadline - 12:
            break
        alt_blocks = greedy_cover(500 + start_seed)
        if len(alt_blocks) < best_count + 15:
            alt_blocks, _ = exact_deletion_pass(alt_blocks, deadline)
            if time.time() < deadline - 8:
                alt_blocks = swap_polish(alt_blocks, deadline)
            if len(alt_blocks) < best_count:
                best_blocks = alt_blocks
                best_count = len(alt_blocks)

    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return [list(b) for b in best_blocks]
