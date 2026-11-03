import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 132.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Step 3: For each block, compute which t-subsets it covers
    block_t_subsets = []
    block_bits = []
    for block in k_blocks:
        covered = set()
        bits = 0
        for combo in itertools.combinations(block, t):
            idx = t_index[combo]
            covered.add(idx)
            bits |= (1 << idx)
        block_t_subsets.append(covered)
        block_bits.append(bits)

    if elapsed() > deadline - 40:
        return [list(b) for b in k_blocks]

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)

    # Step 4: Weighted greedy set cover
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    t_weights = 1.0 / safe_counts

    gain = np.zeros(num_k, dtype=np.float64)
    for i, covered in enumerate(block_t_subsets):
        if covered:
            gain[i] = np.sum(t_weights[list(covered)])

    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)

    while uncovered_count > 0 and elapsed() < deadline - 30:
        masked_gain = gain.copy()
        masked_gain[selected_mask] = -1.0
        best_idx = int(np.argmax(masked_gain))
        best_gain = gain[best_idx]

        if best_gain <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

        for idx in block_t_subsets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = t_weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w

    result = [list(k_blocks[i]) for i in selected]

    # Post-processing 1: Remove redundant blocks
    if elapsed() < deadline - 25 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 25:
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 25:
                    break
                block_idx = selected[idx_remove]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(idx_remove)
                    result = [list(k_blocks[i]) for i in selected]
                    changed = True

    # Post-processing 2: Optimized Local Search with Bitsets
    if elapsed() < deadline - 15 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        selected_set = set(selected)
        improved = True
        max_rounds = 100
        
        for _ in range(max_rounds):
            if elapsed() > deadline - 15:
                break
            if not improved:
                break
            improved = False

            for si in range(len(selected)):
                if elapsed() > deadline - 15:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                
                # Find critical t-subsets
                critical = []
                for idx in drop_set:
                    if cover_count[idx] == 1:
                        critical.append(idx)

                if not critical:
                    # Can drop without replacement
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    break

                # Need replacement
                critical_bits = 0
                for c in critical:
                    critical_bits |= (1 << c)

                best_candidate = -1
                best_score = float('inf')

                # Pre-compute "potential" for candidates? 
                # We want to minimize the number of t-subsets that become "critical" (count 1) after the swap.
                # New count N_idx = C_idx - 1_{idx in A} + 1_{idx in B}
                # We want to minimize sum(1 if N_idx == 1 else 0)
                
                # Iterate candidates
                for b in range(num_k):
                    if b in selected_set:
                        continue
                    # Check if b covers all criticals
                    if (block_bits[b] & critical_bits) != critical_bits:
                        continue
                    
                    # Calculate score: number of idx where new count is 1
                    # N_idx = C_idx - 1_{idx in A} + 1_{idx in B}
                    # We can iterate over union of A and B
                    # But A and B are small (56 elements)
                    # We can just iterate over B and A
                    # Actually, we can just iterate over all idx in B, and check if idx in A
                    # And also check idx in A not in B? No, if idx in A and not B, N_idx = C_idx - 1.
                    # If C_idx - 1 == 1, i.e., C_idx == 2, then it becomes critical.
                    
                    score = 0
                    # Check idx in B
                    for idx in block_t_subsets[b]:
                        in_a = idx in drop_set
                        c_val = cover_count[idx]
                        if in_a:
                            new_c = c_val - 1 + 1
                        else:
                            new_c = c_val + 1
                        if new_c == 1:
                            score += 1
                            
                    # Check idx in A but not in B
                    # We need to know which idx in A are NOT in B
                    # b_set = block_t_subsets[b]
                    for idx in drop_set:
                        if idx not in block_t_subsets[b]:
                            c_val = cover_count[idx]
                            new_c = c_val - 1
                            if new_c == 1:
                                score += 1
                            
                    if score < best_score:
                        best_score = score
                        best_candidate = b

                if best_candidate != -1:
                    # Perform swap
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    break

    # Post-processing 3: Final drops
    if elapsed() < deadline - 10 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 10:
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 10:
                    break
                block_idx = selected[idx_remove]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(idx_remove)
                    result = [list(k_blocks[i]) for i in selected]
                    changed = True

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(selected)
            for i in range(num_k):
                if elapsed() > deadline - 5:
                    break
                if i in selected_set:
                    continue
                has_uncovered = False
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        has_uncovered = True
                        break
                if has_uncovered:
                    selected.append(i)
                    selected_set.add(i)
                    for idx2 in block_t_subsets[i]:
                        final_cover[idx2] += 1
            result = [list(k_blocks[i]) for i in selected]
    elif num_t > 0:
        uncovered = np.ones(num_t, dtype=np.bool_)
        for i in range(num_k):
            if elapsed() > deadline - 5:
                break
            has_uncovered = False
            for idx in block_t_subsets[i]:
                if uncovered[idx]:
                    has_uncovered = True
                    break
            if has_uncovered:
                selected.append(i)
                for idx2 in block_t_subsets[i]:
                    uncovered[idx2] = False
            if not uncovered.any():
                break
        result = [list(k_blocks[i]) for i in selected]

    return result
