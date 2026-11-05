import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Generate all k-subsets (blocks) and their coverage masks
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)

    # Precompute for each block which t-subsets it covers
    block_ts_indices = []
    for block in k_subsets:
        ts_indices = []
        for t_sub in itertools.combinations(block, t):
            ts_indices.append(t_subset_to_idx[t_sub])
        block_ts_indices.append(ts_indices)

    # Build bitmasks for fast coverage checking
    block_masks = []
    for ts_indices in block_ts_indices:
        mask = 0
        for idx in ts_indices:
            mask |= (1 << idx)
        block_masks.append(mask)

    full_mask = (1 << num_t_subsets) - 1

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(noise_scale=0.0):
        """Run greedy set cover with optional random noise for tie-breaking."""
        uncovered = full_mask
        selected = []

        while uncovered:
            best_gain = -1
            best_idx = -1

            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    # Add small random noise to break ties and explore
                    noisy_gain = gain + random.random() * noise_scale
                    if noisy_gain > best_gain:
                        best_gain = noisy_gain
                        best_idx = i

            if best_idx == -1:
                break  # Should not happen if uncovered != 0

            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return selected, True  # True indicates timeout

        return selected, False

    def packing_first_build(seed=0):
        """
        Packing-First strategy:
        Phase 1: Build a high-efficiency core by greedily selecting blocks that
                 maximize NEW coverage (minimizing overlap with already-selected).
                 This creates a structurally efficient packing.
        Phase 2: Repair remaining uncovered t-subsets with targeted additions.
        """
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        # Phase 1: Packing-first core build
        # We want to maximize new coverage per block, which naturally minimizes overlap.
        # Use a slightly different noise profile to encourage diversity in the core.
        core_noise_scales = [0.0, 0.3, 0.7, 1.5]
        core_noise = core_noise_scales[seed % len(core_noise_scales)]

        while uncovered:
            best_gain = -1
            best_idx = -1
            best_noisy = -1

            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noisy_gain = gain + rng.random() * core_noise
                    if noisy_gain > best_noisy:
                        best_noisy = noisy_gain
                        best_gain = gain
                        best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return selected, True

        # Phase 2: Should not have uncovered if all t-subsets are covered by some block,
        # but due to the greedy nature, we verify and repair if needed.
        # In practice, for C(19,7,4), every 4-subset is contained in some 7-subset,
        # so the greedy should cover everything. But let's be safe.
        if uncovered:
            # Repair phase: add blocks to cover remaining gaps
            while uncovered:
                best_gain = -1
                best_idx = -1
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected.append(best_idx)
                selected_set.add(best_idx)
                uncovered &= ~block_masks[best_idx]
                if time.time() > deadline:
                    return selected, True

        return selected, False

    def remove_redundant(selected):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not selected:
            return selected

        # Count how many times each t-subset is covered
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        active = [True] * len(selected)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                # Check if this block has any uniquely covered t-subset
                has_unique = False
                for ts_idx in block_ts_indices[selected[i]]:
                    if counts[ts_idx] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    active[i] = False
                    changed = True
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1

        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search_swap(selected, max_swaps=500):
        # (kept as-is; SA polish is added separately below)
        """Try to swap blocks to enable removal of other blocks."""
        if not selected:
            return selected

        selected = list(selected)
        n = len(selected)

        # Build counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected

                # Find t-subsets uniquely covered by block i
                unique_ts = []
                for ts_idx in block_ts_indices[selected[i]]:
                    if counts[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    # Can remove block i directly
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1
                    selected.pop(i)
                    improved = True
                    swap_count += 1
                    break

                # Try to find a replacement block that covers all unique_ts
                # and allows removal of another block
                for j in range(len(selected)):
                    if j == i:
                        continue

                    # Check if block j covers all unique t-subsets of block i
                    covers_all = True
                    for ts_idx in unique_ts:
                        if ts_idx not in block_ts_indices[selected[j]]:
                            covers_all = False
                            break

                    if not covers_all:
                        continue

                    # Simulate: remove block i, add block j
                    # Then check if we can remove any other block

                    # Temporarily update counts
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_ts_indices[selected[j]]:
                        counts[ts_idx] += 1

                    # Check if any block (other than i and j) is now redundant
                    removable = -1
                    for m in range(len(selected)):
                        if m == i or m == j:
                            continue
                        is_redundant = True
                        for ts_idx in block_ts_indices[selected[m]]:
                            if counts[ts_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            removable = m
                            break

                    if removable != -1:
                        # Perform the swap and removal
                        # Remove block i
                        selected.pop(i)
                        # Now j's index may have shifted
                        if j > i:
                            j -= 1
                        # Remove the redundant block
                        if removable > j:
                            removable -= 1
                        for ts_idx in block_ts_indices[selected[removable]]:
                            counts[ts_idx] -= 1
                        selected.pop(removable)
                        improved = True
                        swap_count += 1
                        break
                    else:
                        # Revert the temporary count changes
                        for ts_idx in block_ts_indices[selected[i]]:
                            counts[ts_idx] += 1
                        for ts_idx in block_ts_indices[selected[j]]:
                            counts[ts_idx] -= 1
                    if improved:
                        break

                if improved:
                    break

        return selected

    def sa_polish(selected, time_budget=8.0, seed=42):
        """Simulated Annealing polish: swap blocks to reduce count.
        
        State: set of selected block indices.
        Energy: number of blocks (lower is better).
        Moves: 
          1. Remove a block if it doesn't leave any t-subset uncovered.
          2. Swap a block with another that allows removal of a redundant block.
        Acceptance: Metropolis criterion with cooling.
        """
        rng = random.Random(seed)
        selected = list(selected)
        n = len(selected)
        if n < 2:
            return selected

        # Build coverage counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        def is_valid():
            return np.all(counts > 0)

        # Precompute: for each selected block, which t-subsets it uniquely covers
        def unique_ts_for(block_pos):
            """Return list of t-subset indices uniquely covered by selected[block_pos]."""
            return [ts_idx for ts_idx in block_ts_indices[selected[block_pos]] if counts[ts_idx] == 1]

        def can_remove(block_pos):
            """Check if removing selected[block_pos] leaves all t-subsets covered."""
            for ts_idx in block_ts_indices[selected[block_pos]]:
                if counts[ts_idx] == 1:
                    return False
            return True

        def do_remove(block_pos):
            """Remove selected[block_pos] and update counts."""
            for ts_idx in block_ts_indices[selected[block_pos]]:
                counts[ts_idx] -= 1
            selected.pop(block_pos)

        def do_add(block_idx, pos=None):
            """Add block_idx to selected at position pos (or end)."""
            if pos is None:
                selected.append(block_idx)
            else:
                selected.insert(pos, block_idx)
            for ts_idx in block_ts_indices[block_idx]:
                counts[ts_idx] += 1

        best_selected = list(selected)
        best_energy = len(selected)
        current_energy = len(selected)

        # Cooling schedule
        T0 = 2.0
        T_min = 0.01
        cooling_rate = 0.995

        T = T0
        steps = 0
        max_steps = 500000

        sa_start = time.time()

        while T > T_min and steps < max_steps:
            if time.time() - sa_start > time_budget:
                break

            steps += 1

            # Generate a random move
            move_type = rng.random()
            if move_type < 0.4 and len(selected) > 1:
                # Try to remove a random block
                pos = rng.randint(0, len(selected) - 1)
                if can_remove(pos):
                    # Always accept improving move
                    do_remove(pos)
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_selected = list(selected)
                else:
                    # Block has unique t-subsets; try swap
                    unique_ts = unique_ts_for(pos)
                    if not unique_ts:
                        do_remove(pos)
                        current_energy -= 1
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_selected = list(selected)
                    else:
                        # Find a candidate block that covers all unique_ts
                        # and after swap, some other block becomes removable
                        # Sample a few candidates
                        found_swap = False
                        for _ in range(3):
                            cand_idx = rng.randint(0, num_blocks - 1)
                            if cand_idx in selected:
                                continue
                            # Check if cand covers all unique_ts
                            cand_ts_set = set(block_ts_indices[cand_idx])
                            if all(ts in cand_ts_set for ts in unique_ts):
                                # Simulate: remove pos, add cand, check if any other block is removable
                                # Temporarily update
                                for ts_idx in block_ts_indices[selected[pos]]:
                                    counts[ts_idx] -= 1
                                for ts_idx in block_ts_indices[cand_idx]:
                                    counts[ts_idx] += 1

                                # Check if any block (except pos and cand) is now removable
                                removable = -1
                                for m in range(len(selected)):
                                    if m == pos:
                                        continue
                                    rem_ok = True
                                    for ts_idx in block_ts_indices[selected[m]]:
                                        if counts[ts_idx] == 1:
                                            rem_ok = False
                                            break
                                    if rem_ok:
                                        removable = m
                                        break

                                if removable != -1:
                                    # Perform the move: remove pos, add cand, remove removable
                                    # First, physically remove pos
                                    block_at_pos = selected.pop(pos)
                                    # Remove removable (adjust index if needed)
                                    if removable > pos:
                                        removable -= 1
                                    rem_block = selected.pop(removable)
                                    for ts_idx in block_ts_indices[rem_block]:
                                        counts[ts_idx] -= 1
                                    # Add cand
                                    do_add(cand_idx)
                                    current_energy -= 1  # removed 2, added 1
                                    if current_energy < best_energy:
                                        best_energy = current_energy
                                        best_selected = list(selected)
                                    found_swap = True
                                    break
                                else:
                                    # Revert
                                    for ts_idx in block_ts_indices[selected[pos]]:
                                        counts[ts_idx] += 1
                                    for ts_idx in block_ts_indices[cand_idx]:
                                        counts[ts_idx] -= 1

                        if found_swap:
                            continue

            elif move_type < 0.7:
                # Swap a random block with a random other block
                if len(selected) < 2:
                    T *= cooling_rate
                    continue
                pos1 = rng.randint(0, len(selected) - 1)
                pos2 = rng.randint(0, len(selected) - 1)
                if pos1 == pos2:
                    T *= cooling_rate
                    continue

                # Check if swapping pos1 and pos2 is valid (both still covered)
                # After swap, we need all t-subsets still covered
                # This is complex; try a simpler approach:
                # Remove pos1, see if we can replace it with a block that makes pos2 removable
                
                unique_ts1 = unique_ts_for(pos1)
                if not unique_ts1:
                    # Can remove pos1 directly
                    do_remove(pos1)
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_selected = list(selected)
                    continue

                # Try to find a replacement for pos1 that covers unique_ts1
                # and makes pos2 removable
                found = False
                for _ in range(3):
                    cand_idx = rng.randint(0, num_blocks - 1)
                    if cand_idx in selected:
                        continue
                    cand_ts_set = set(block_ts_indices[cand_idx])
                    if not all(ts in cand_ts_set for ts in unique_ts1):
                        continue

                    # Simulate
                    for ts_idx in block_ts_indices[selected[pos1]]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_ts_indices[cand_idx]:
                        counts[ts_idx] += 1

                    # Check if pos2 is now removable
                    pos2_removable = True
                    for ts_idx in block_ts_indices[selected[pos2]]:
                        if counts[ts_idx] == 1:
                            pos2_removable = False
                            break

                    if pos2_removable:
                        # Perform: remove pos1, add cand, remove pos2
                        block_at_pos1 = selected.pop(pos1)
                        if pos2 > pos1:
                            pos2 -= 1
                        rem_block = selected.pop(pos2)
                        for ts_idx in block_ts_indices[rem_block]:
                            counts[ts_idx] -= 1
                        do_add(cand_idx)
                        current_energy -= 1
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_selected = list(selected)
                        found = True
                        break
                    else:
                        # Revert
                        for ts_idx in block_ts_indices[selected[pos1]]:
                            counts[ts_idx] += 1
                        for ts_idx in block_ts_indices[cand_idx]:
                            counts[ts_idx] -= 1

                if found:
                    continue

            else:
                # Add a random block (worsening move, accepted with probability)
                if len(selected) >= 150:
                    T *= cooling_rate
                    continue
                cand_idx = rng.randint(0, num_blocks - 1)
                if cand_idx in selected:
                    T *= cooling_rate
                    continue

                # Add it
                do_add(cand_idx)
                new_energy = len(selected)
                delta_e = new_energy - current_energy  # +1

                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                else:
                    # Revert
                    selected.pop()
                    for ts_idx in block_ts_indices[cand_idx]:
                        counts[ts_idx] -= 1

            # Cool down
            T *= cooling_rate

        return best_selected

    # Run multiple strategies: packing-first with different seeds, plus original greedy
    # Packing-first first (3 variants), then original greedy (3 variants)
    strategies = [
        ('packing', 0), ('packing', 1), ('packing', 2),
        ('greedy', 0.0), ('greedy', 0.5), ('greedy', 1.0),
    ]

    for strat, param in strategies:
        if time.time() > deadline:
            break

        if strat == 'packing':
            selected, timed_out = packing_first_build(param)
        else:
            selected, timed_out = greedy_cover(param)

        if timed_out:
            continue

        # Remove redundant blocks
        selected = remove_redundant(selected)

        # Local search
        selected = local_search_swap(selected)

        # Final redundancy removal
        selected = remove_redundant(selected)

        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [k_subsets[i] for i in selected]

            if best_count <= 131:  # Lower bound
                break

    # SA polish on the best solution found
    if best_blocks is not None and time.time() < deadline:
        # Convert back to indices for SA
        block_to_idx = {block: i for i, block in enumerate(k_subsets)}
        selected_indices = [block_to_idx[b] for b in best_blocks]
        
        # Run SA with multiple seeds, using remaining time
        sa_time_remaining = deadline - time.time() - 5  # 5s safety margin
        if sa_time_remaining > 2:
            sa_budget = min(sa_time_remaining, 12)
            for sa_seed in [42, 123, 777]:
                if time.time() > deadline - 5:
                    break
                polished = sa_polish(selected_indices, time_budget=sa_budget / 3.0, seed=sa_seed)
                polished = remove_redundant(polished)
                if len(polished) < best_count:
                    best_count = len(polished)
                    best_blocks = [k_subsets[i] for i in polished]
                    if best_count <= 131:
                        break

    if best_blocks is None:
        # Fallback: just run deterministic greedy
        selected, _ = greedy_cover(0.0)
        selected = remove_redundant(selected)
        best_blocks = [k_subsets[i] for i in selected]

    return best_blocks
