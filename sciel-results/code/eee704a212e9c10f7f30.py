import itertools
import random
import time
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)

    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)

    k_to_t_indices = []
    t_to_k_indices = [[] for _ in range(num_t)]

    for ki, k_sub in enumerate(all_k_subsets):
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            ti = t_subset_to_idx[t_sub]
            t_indices.append(ti)
            t_to_k_indices[ti].append(ki)
        k_to_t_indices.append(t_indices)

    block_masks = [0] * num_k
    for ki, t_indices in enumerate(k_to_t_indices):
        mask = 0
        for ti in t_indices:
            mask |= (1 << ti)
        block_masks[ki] = mask

    def run_greedy(seed=None):
        rng = random.Random(seed) if seed is not None else random.Random()

        uncovered_mask = (1 << num_t) - 1
        uncovered_count = num_t
        selected_blocks = []

        heap = []
        for ki in range(num_k):
            gain = (uncovered_mask & block_masks[ki]).bit_count()
            if gain > 0:
                noise = rng.random() * 0.01
                heapq.heappush(heap, (-(gain + noise), ki))

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            best_ki = -1

            while heap:
                neg_gain, ki = heapq.heappop(heap)
                actual_gain = (uncovered_mask & block_masks[ki]).bit_count()
                if actual_gain == 0:
                    continue
                if actual_gain != int(-neg_gain):
                    noise = rng.random() * 0.01
                    heapq.heappush(heap, (-(actual_gain + noise), ki))
                    continue
                best_ki = ki
                break

            if best_ki == -1:
                break

            selected_blocks.append(best_ki)
            block_mask = block_masks[best_ki]
            newly_covered = uncovered_mask & block_mask
            if newly_covered:
                uncovered_count -= newly_covered.bit_count()
                uncovered_mask &= ~newly_covered

        return selected_blocks

    def local_search(blocks_list, time_budget=5.0):
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        end_time = time.time() + time_budget

        def compute_counts(blks):
            counts = [0] * num_t
            for ki in blks:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            return counts

        improved = True
        while improved and time.time() < end_time:
            improved = False
            counts = compute_counts(current_blocks)

            # Remove blocks with no unique subsets
            removed = False
            for i in range(len(current_blocks)):
                ki = current_blocks[i]
                has_unique = any(counts[ti] == 1 for ti in k_to_t_indices[ki])
                if not has_unique:
                    current_blocks.pop(i)
                    current_set.discard(ki)
                    improved = True
                    removed = True
                    break
            if removed:
                continue

            # Single-block replacement
            for i in range(len(current_blocks)):
                ki = current_blocks[i]
                unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                if not unique_subs:
                    continue

                if len(unique_subs) == 1:
                    ti = unique_subs[0]
                    for new_ki in t_to_k_indices[ti]:
                        if new_ki == ki or new_ki in current_set:
                            continue
                        current_blocks[i] = new_ki
                        current_set.discard(ki)
                        current_set.add(new_ki)
                        improved = True
                        break
                else:
                    us_sorted = sorted(unique_subs, key=lambda ti: len(t_to_k_indices[ti]))
                    candidates = set(t_to_k_indices[us_sorted[0]])
                    for ti in us_sorted[1:]:
                        candidates &= set(t_to_k_indices[ti])
                        if not candidates:
                            break
                    candidates.discard(ki)
                    candidates -= current_set
                    if candidates:
                        new_ki = random.choice(list(candidates))
                        current_blocks[i] = new_ki
                        current_set.discard(ki)
                        current_set.add(new_ki)
                        improved = True
                        break

                if improved:
                    break

            # 2-exchange: replace two blocks simultaneously
            if not improved and len(current_blocks) >= 2:
                blocks_with_unique = []
                for i, ki in enumerate(current_blocks):
                    unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                    if unique_subs:
                        blocks_with_unique.append((i, ki, unique_subs))

                if len(blocks_with_unique) >= 2:
                    for a in range(min(len(blocks_with_unique), 4)):
                        for b in range(a + 1, min(len(blocks_with_unique), 4)):
                            ia, ki_a, unique_a = blocks_with_unique[a]
                            ib, ki_b, unique_b = blocks_with_unique[b]
                            all_unique = unique_a + unique_b
                            au_sorted = sorted(all_unique, key=lambda ti: len(t_to_k_indices[ti]))
                            cand_a = set(t_to_k_indices[au_sorted[0]])
                            cand_b = set(t_to_k_indices[au_sorted[0]])
                            for ti in au_sorted[1:]:
                                cand_a &= set(t_to_k_indices[ti])
                                cand_b &= set(t_to_k_indices[ti])
                                if not cand_a or not cand_b:
                                    break
                            cand_a.discard(ki_a)
                            cand_a.discard(ki_b)
                            cand_b.discard(ki_a)
                            cand_b.discard(ki_b)
                            cand_a -= current_set
                            cand_b -= current_set
                            if cand_a and cand_b:
                                new_a = random.choice(list(cand_a))
                                new_b = random.choice(list(cand_b))
                                if new_a != new_b:
                                    current_blocks[ia] = new_a
                                    current_blocks[ib] = new_b
                                    current_set.discard(ki_a)
                                    current_set.discard(ki_b)
                                    current_set.add(new_a)
                                    current_set.add(new_b)
                                    improved = True
                                    break
                        if improved:
                            break

        return current_blocks

    best_blocks = None
    best_count = float('inf')
    max_restarts = 50

    for seed in range(max_restarts):
        if time.time() > deadline - 15:
            break

        blocks_idx = run_greedy(seed)
        if not blocks_idx:
            continue

        # Local search with time budget
        ls_time = min(8.0, max(2.0, (deadline - 5 - time.time()) * 0.3))
        result = local_search(blocks_idx, ls_time)

        if len(result) < best_count:
            best_count = len(result)
            best_blocks = result

    if best_blocks is None:
        blocks_idx = run_greedy(0)
        if blocks_idx:
            best_blocks = blocks_idx
        else:
            return []

    return [list(all_k_subsets[ki]) for ki in best_blocks]
