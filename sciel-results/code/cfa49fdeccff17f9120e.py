"""Covering design C(19,8,5): greedy + local search with bitmask acceleration."""
import itertools
import random
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # Safety margin under 150s limit

    # Precomputation
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    n_blocks = len(all_blocks)

    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)

    # For each block, compute the bitmask of t-subsets it covers
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Greedy set cover
    uncovered_mask = (1 << num_subsets) - 1
    selected = []
    counts = [0] * num_subsets

    while uncovered_mask and time.time() < deadline:
        best_block_idx = -1
        best_gain = 0
        for i in range(n_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i

        if best_block_idx == -1 or best_gain == 0:
            break

        selected.append(best_block_idx)
        mask = block_masks[best_block_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
        uncovered_mask &= ~mask

    # Remove redundant blocks
    def remove_redundant(sol):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_selected = []
            for b_idx in sol:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt = 0
                    for other in sol:
                        if other != b_idx and (block_masks[other] >> idx) & 1:
                            cnt += 1
                            if cnt >= 1:
                                break
                    if cnt == 0:
                        redundant = False
                        break
                    m ^= lsb

                if redundant:
                    changed = True
                else:
                    new_selected.append(b_idx)
            sol = new_selected
        return sol

    selected = remove_redundant(selected)
    best_solution = list(selected)
    best_len = len(selected)

    # Local search: try to remove blocks and replace with better ones
    random.seed(42)
    current = list(best_solution)

    # Helper: check if removing a block still leaves all subsets covered
    def can_remove(sol, b_idx):
        """Check if b_idx can be removed from sol while maintaining coverage."""
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cnt = 0
            for other in sol:
                if other != b_idx and (block_masks[other] >> idx) & 1:
                    cnt += 1
                    break
            if cnt == 0:
                return False
            m ^= lsb
        return True

    # Local search loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        cur_len = len(current)
        
        for i in range(cur_len):
            if time.time() > deadline - 10:
                break
            
            # Try removing block i
            if can_remove(current, i):
                new_sol = current[:i] + current[i+1:]
                current = new_sol
                improved = True
                if len(current) < best_len:
                    best_solution = list(current)
                    best_len = len(current)
                break  # Restart from beginning after a removal
        
        # If no removal possible, try swaps
        if not improved and time.time() < deadline - 10:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                old_block = current[i]
                # Try replacing with a random block
                for _ in range(50):
                    new_block = random.randint(0, n_blocks - 1)
                    if new_block == old_block:
                        continue
                    test_sol = current[:]
                    test_sol[i] = new_block
                    # Check validity
                    covered = [0] * num_subsets
                    for b_idx in test_sol:
                        mask = block_masks[b_idx]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            covered[idx] += 1
                            m ^= lsb
                    valid = all(c > 0 for c in covered)
                    if valid:
                        current = test_sol
                        improved = True
                        break
                if improved:
                    break
        
        # Try adding a block then removing multiple
        if not improved and time.time() < deadline - 10 and len(current) < best_len + 20:
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                new_block = random.randint(0, n_blocks - 1)
                if new_block not in current:
                    test_sol = current + [new_block]
                    # Try to remove blocks from test_sol
                    test_sol = remove_redundant(test_sol)
                    if len(test_sol) < len(current):
                        current = test_sol
                        improved = True
                        if len(current) < best_len:
                            best_solution = list(current)
                            best_len = len(current)
                        break

    # Final cleanup
    best_solution = remove_redundant(best_solution)

    result = [list(all_blocks[i]) for i in best_solution]
    return result
