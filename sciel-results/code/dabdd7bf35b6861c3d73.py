"""Combined program: fast greedy construction + local search (remove, 2-for-1, 1-for-1) + max-flow refinement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import deque


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute (t-1)-subset to t-subsets mapping
    # For t=3: pair -> list of triplets containing that pair
    t1_to_tsets = {}
    for tid, ts in enumerate(all_tsets):
        for sub in itertools.combinations(ts, t - 1):
            t1_to_tsets.setdefault(sub, []).append(tid)

    def get_id(ts_tuple):
        return tset_to_id[ts_tuple]

    def greedy_construct(seed=0, strategy='max_gain'):
        """Greedy construction with different strategies.
        strategy: 'max_gain' (absolute), 'ratio' (gain/1, same for fixed k),
                  'ratio2' (gain / overlap_penalty), 'random' (randomized)
        """
        rng = random.Random(seed)
        uncovered_arr = np.ones(n_tsets, dtype=np.int8)
        blocks = []

        while uncovered_arr.any():
            # Pick a random uncovered t-set to start
            uncov_indices = np.where(uncovered_arr == 1)[0]
            start_id = uncov_indices[rng.randrange(len(uncov_indices))]
            start_ts = all_tsets[start_id]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                best_score = -1
                candidates = [p for p in range(v) if p not in block_set]

                subs_in_block = list(itertools.combinations(block, t - 1))

                # Vectorized gain computation
                # For each candidate p and each (t-1)-subset, compute the t-set id
                gains = np.zeros(len(candidates), dtype=np.int32)
                for sub in subs_in_block:
                    for ci, p in enumerate(candidates):
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if uncovered_arr[tid]:
                            gains[ci] += 1

                if strategy == 'max_gain':
                    scores = gains.astype(np.float64)
                elif strategy == 'ratio':
                    denom = 1 + 0.1 * (k - len(block) - 1)
                    scores = gains.astype(np.float64) / denom
                elif strategy == 'ratio2':
                    progress = len(block) / k
                    scores = gains.astype(np.float64) / (1 + progress)
                else:  # random
                    scores = np.array([rng.random() * g if g > 0 else rng.random() * 0.1 for g in gains])

                # Add small random tie-breaker for diversity
                scores += rng.random() * 0.01

                best_idx = int(np.argmax(scores))
                best_score = scores[best_idx]
                best_point = candidates[best_idx]

                if best_score <= 0.01:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                uncovered_arr[tid] = 0

        return blocks

    # Phase 1: Multiple greedy restarts with varied strategies
    best_blocks = None
    best_count = float('inf')

    strategies = ['max_gain', 'max_gain', 'ratio', 'ratio2', 'random']
    num_seeds = 40
    for seed in range(num_seeds):
        if time.time() > deadline - 25:
            break
        strategy = strategies[seed % len(strategies)]
        blocks = greedy_construct(seed, strategy)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0, 'max_gain')

    # Phase 2: Local search
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    # Max-flow based minimum set cover refinement
    def maxflow_refine(blocks):
        """Use max-flow to find minimum subset of blocks that covers all t-sets.
        Build bipartite graph: source -> blocks -> t-sets -> sink.
        Find min cut; blocks on source side of min cut form minimum cover.
        """
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks

        # Build adjacency: block -> list of t-set ids it covers
        block_to_tsets = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(get_id(ts))
            block_to_tsets.append(ids)

        # Only t-sets that are actually needed (all should be, but check)
        # Build bipartite graph for max-flow
        # Nodes: source=0, blocks=1..n_blocks, t-sets=n_blocks+1..n_blocks+n_tsets, sink=n_blocks+n_tsets+1
        n_nodes = 1 + n_blocks + n_tsets + 1
        source = 0
        sink = n_nodes - 1

        # Adjacency list for residual graph
        # Use arrays for speed
        adj = [[] for _ in range(n_nodes)]
        # Edge list: (from, to, capacity)
        edge_list = []

        def add_edge(u, v, cap):
            edge_list.append([u, v, cap, len(adj[u]), len(adj[v])])
            adj[u].append(len(edge_list) - 1)
            edge_list.append([v, u, 0, len(adj[v]), len(adj[u])])
            adj[v].append(len(edge_list) - 1)

        # Source to blocks: capacity 1
        for i in range(n_blocks):
            add_edge(source, 1 + i, 1)

        # Blocks to t-sets: capacity 1 (infinite would work too, but 1 is sufficient since we want min cardinality)
        for i in range(n_blocks):
            for tid in block_to_tsets[i]:
                add_edge(1 + i, 1 + n_blocks + tid, 1)

        # T-sets to sink: capacity 1 (demand)
        for j in range(n_tsets):
            add_edge(1 + n_blocks + j, sink, 1)

        # Dinic's algorithm
        def dinic_maxflow():
            flow = 0
            n = n_nodes
            level = [-1] * n
            ptr = [0] * n

            def bfs():
                for i in range(n):
                    level[i] = -1
                q = deque([source])
                level[source] = 0
                while q:
                    u = q.popleft()
                    for ei in adj[u]:
                        e = edge_list[ei]
                        if e[2] > 0 and level[e[1]] == -1:
                            level[e[1]] = level[u] + 1
                            q.append(e[1])
                return level[sink] != -1

            def dfs(u, pushed):
                if pushed == 0:
                    return 0
                if u == sink:
                    return pushed
                while ptr[u] < len(adj[u]):
                    ei = adj[u][ptr[u]]
                    e = edge_list[ei]
                    if level[e[1]] == level[u] + 1 and e[2] > 0:
                        tr = dfs(e[1], min(pushed, e[2]))
                        if tr == 0:
                            ptr[u] += 1
                        else:
                            e[2] -= tr
                            edge_list[e[3]][2] += tr
                            return tr
                    else:
                        ptr[u] += 1
                return 0

            while bfs():
                for i in range(n):
                    ptr[i] = 0
                while True:
                    pushed = dfs(source, float('inf'))
                    if pushed == 0:
                        break
                    flow += pushed
            return flow

        max_flow = dinic_maxflow()

        # Find min cut: BFS from source in residual graph
        visited = [False] * n_nodes
        q = deque([source])
        visited[source] = True
        while q:
            u = q.popleft()
            for ei in adj[u]:
                e = edge_list[ei]
                if e[2] > 0 and not visited[e[1]]:
                    visited[e[1]] = True
                    q.append(e[1])

        # Blocks on source side of min cut are the minimum cover
        min_cover_blocks = []
        for i in range(n_blocks):
            if visited[1 + i]:
                min_cover_blocks.append(blocks[i])

        return min_cover_blocks

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[get_id(ts)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute block t-set IDs for all blocks
        block_ids = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(get_id(ts))
            block_ids.append(ids)

        for i in range(n):
            if time.time() > deadline - 10:
                return blocks, False
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    return blocks, False

                b1 = blocks[i]
                b2 = blocks[j]
                ids_b1 = block_ids[i]
                ids_b2 = block_ids[j]

                check_ids = ids_b1 | ids_b2

                # Find t-sets uniquely covered by b1 or b2
                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need.append(tid)

                if not u_need:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                # Find required points
                p_need = set()
                for tid in u_need:
                    p_need.update(all_tsets[tid])

                if len(p_need) > k:
                    continue

                # Try multiple strategies to build replacement block
                best_new_block = None
                best_additional_gain = -1
                rng = random.Random(i * 1000 + j)

                for attempt in range(3):
                    new_block = list(p_need)
                    new_block_set = set(new_block)
                    remaining_slots = k - len(new_block)
                    candidates = [p for p in range(v) if p not in new_block_set]

                    for _ in range(remaining_slots):
                        if not candidates:
                            break
                        best_p = -1
                        best_gain = -1
                        subs_in_block = list(itertools.combinations(new_block, t - 1))

                        for p in candidates:
                            gain = 0
                            for sub in subs_in_block:
                                ts = tuple(sorted(sub + (p,)))
                                tid = get_id(ts)
                                if tid not in ids_b1 and tid not in ids_b2:
                                    gain += 1
                            # Add randomness for diversity across attempts
                            if attempt > 0:
                                gain += rng.random() * 0.5
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p

                        if best_p == -1 or best_gain == 0:
                            best_p = rng.choice(candidates)

                        new_block.append(best_p)
                        new_block_set.add(best_p)
                        candidates.remove(best_p)

                    new_block.sort()

                    # Verify the new block actually covers all u_need
                    new_ids = set()
                    for ts in itertools.combinations(new_block, t):
                        new_ids.add(get_id(ts))

                    if all(tid in new_ids for tid in u_need):
                        # Count additional coverage beyond u_need
                        addl = len(new_ids - ids_b1 - ids_b2)
                        if addl > best_additional_gain:
                            best_additional_gain = addl
                            best_new_block = new_block

                if best_new_block is not None:
                    update_counts([b1, b2], -1)
                    update_counts([best_new_block], 1)
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(best_new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 10:
                return blocks, False

            b = blocks[i]

            # Get t-set IDs covered by b
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

            # Find t-sets uniquely covered by b
            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(tid)

            if not u_need:
                # Block is redundant
                update_counts([b], -1)
                return [x for idx, x in enumerate(blocks) if idx != i], True

            # Find required points
            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            # Build new block optimizing for additional coverage
            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 7919)
            candidates = [p for p in range(v) if p not in new_block_set]

            remaining_slots = k - len(new_block)
            for _ in range(remaining_slots):
                if not candidates:
                    break
                best_p = -1
                best_gain = -1
                subs_in_block = list(itertools.combinations(new_block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid not in ids_b:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p

                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)

                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)

            new_block.sort()

            # Verify coverage and ensure different from old
            if new_block != b:
                new_ids = set()
                for ts in itertools.combinations(new_block, t):
                    new_ids.add(get_id(ts))

                if all(tid in new_ids for tid in u_need):
                    update_counts([b], -1)
                    update_counts([new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves with multiple passes per round
    improved = True
    max_rounds = 150
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if not improved:
            break
        improved = False

        # Try removing redundant blocks (multiple passes)
        for _pass in range(3):
            if time.time() > deadline - 15:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        # Try 2-for-1 swaps
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            # After a swap, try more removals
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        # Try 1-for-1 swaps
        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            # After a swap, try more removals
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

    # Phase 3: Max-flow refinement to find minimum subset of blocks
    if time.time() < deadline - 20:
        mf_blocks = maxflow_refine(current_blocks)
        if len(mf_blocks) < len(current_blocks):
            current_blocks = mf_blocks
            # Update counts after max-flow refinement
            count.fill(0)
            update_counts(current_blocks, 1)
            # Try more local search after max-flow
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                sub_improved = False
                for _pass in range(2):
                    if time.time() > deadline - 10:
                        break
                    new_blocks, rem = try_remove_redundant(current_blocks)
                    if rem:
                        current_blocks = new_blocks
                        sub_improved = True
                    else:
                        break
                new_blocks, swp = try_swap_2_for_1(current_blocks)
                if swp:
                    current_blocks = new_blocks
                    sub_improved = True
                    for _pass in range(2):
                        if time.time() > deadline - 10:
                            break
                        new_blocks, rem = try_remove_redundant(current_blocks)
                        if rem:
                            current_blocks = new_blocks
                            sub_improved = True
                        else:
                            break
                new_blocks, swp = try_swap_1_for_1(current_blocks)
                if swp:
                    current_blocks = new_blocks
                    sub_improved = True
                    for _pass in range(2):
                        if time.time() > deadline - 10:
                            break
                        new_blocks, rem = try_remove_redundant(current_blocks)
                        if rem:
                            current_blocks = new_blocks
                            sub_improved = True
                        else:
                            break
                if not sub_improved:
                    break

    # Final verification: ensure all t-sets are covered
    # (Should be valid, but just in case)
    covered = set()
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(get_id(ts))

    if len(covered) < n_tsets:
        # Fallback: use the best greedy result
        return best_blocks

    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in current_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
# EVOLVE-BLOCK-END
