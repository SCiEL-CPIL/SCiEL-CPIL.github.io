<<<<<<< SEARCH
    # 2. Local Search - aggressive deletion and swap
    no_improvement_count = 0
    max_no_improvement = 100

    while time.time() < deadline - 2:
        improved = False

        # Phase 1: Delete all redundant blocks at once
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)

        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True

        if improved:
            no_improvement_count = 0
            continue

        # Phase 2: Try swaps for critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if np.any(cover_count[block_to_tsubs_np[b_idx]] == 1):
                critical_blocks.append(b_idx)

        if not critical_blocks:
            break

        # Shuffle critical blocks to avoid bias
        random.shuffle(critical_blocks)

        swapped = False
        for b_old_idx in critical_blocks:
            if time.time() > deadline - 2:
                break

            unique_tsubs = block_to_tsubs_np[b_old_idx][cover_count[block_to_tsubs_np[b_old_idx]] == 1]
            if len(unique_tsubs) == 0:
                continue

            # Find blocks that cover all unique t-subsets
            # Use set intersection
            candidate_set = set(t_sub_to_blocks[unique_tsubs[0]])
            for t_idx in unique_tsubs[1:]:
                candidate_set.intersection_update(t_sub_to_blocks[t_idx])
                if not candidate_set:
                    break

            if candidate_set:
                candidate_set.discard(b_old_idx)
                if candidate_set:
                    # Pick a candidate that maximizes additional coverage
                    best_new = None
                    best_score = -1
                    for b_new_idx in list(candidate_set)[:20]:  # Limit to 20 for speed
                        score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                        if score > best_score:
                            best_score = score
                            best_new = b_new_idx

                    if best_new is not None:
                        remove_block(b_old_idx)
                        add_block(best_new)
                        improved = True
                        swapped = True
                        break

        if swapped:
            no_improvement_count = 0
            continue

        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            break

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True
=======
    # 2. Local Search - aggressive deletion and multi-block swaps
    no_improvement_count = 0
    max_no_improvement = 200

    def get_unique_tsubs_for_block(b_idx):
        """Get t-subsets that are uniquely covered by this block."""
        b_tsubs = block_to_tsubs_np[b_idx]
        return b_tsubs[cover_count[b_tsubs] == 1]

    def find_blocks_covering_tsub_list(tsub_list):
        """Find all blocks that cover every t-subset in the list."""
        if not tsub_list:
            return set()
        candidate_set = set(t_sub_to_blocks[int(tsub_list[0])])
        for t_idx in tsub_list[1:]:
            candidate_set.intersection_update(t_sub_to_blocks[int(t_idx)])
            if not candidate_set:
                break
        return candidate_set

    while time.time() < deadline - 5:
        improved = False

        # Phase 1: Delete all redundant blocks at once
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)

        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True

        if improved:
            no_improvement_count = 0
            continue

        # Phase 2: Try 1-for-1 swaps for critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if np.any(cover_count[block_to_tsubs_np[b_idx]] == 1):
                critical_blocks.append(b_idx)

        if not critical_blocks:
            break

        random.shuffle(critical_blocks)

        swapped = False
        for b_old_idx in critical_blocks:
            if time.time() > deadline - 5:
                break

            unique_tsubs = get_unique_tsubs_for_block(b_old_idx)
            if len(unique_tsubs) == 0:
                continue

            candidate_set = find_blocks_covering_tsub_list(unique_tsubs)
            if candidate_set:
                candidate_set.discard(b_old_idx)
                if candidate_set:
                    # Pick a candidate that maximizes additional coverage
                    best_new = None
                    best_score = -1
                    for b_new_idx in list(candidate_set)[:30]:
                        score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                        if score > best_score:
                            best_score = score
                            best_new = b_new_idx

                    if best_new is not None:
                        remove_block(b_old_idx)
                        add_block(best_new)
                        improved = True
                        swapped = True
                        break

        if swapped:
            no_improvement_count = 0
            continue

        # Phase 3: Try 2-for-1 swaps (remove 2 blocks, add 1)
        if time.time() < deadline - 5:
            # Find pairs of blocks whose unique t-subsets can be covered by a single new block
            all_critical = [b for b in blocks_in_cover if np.any(cover_count[block_to_tsubs_np[b]] == 1)]
            random.shuffle(all_critical)

            pair_swapped = False
            for i in range(len(all_critical)):
                if time.time() > deadline - 5:
                    break
                b1 = all_critical[i]
                for j in range(i+1, len(all_critical)):
                    if time.time() > deadline - 5:
                        break
                    b2 = all_critical[j]

                    u1 = get_unique_tsubs_for_block(b1)
                    u2 = get_unique_tsubs_for_block(b2)
                    if len(u1) == 0 or len(u2) == 0:
                        continue

                    # Combined unique t-subsets that must be covered
                    combined = np.concatenate([u1, u2])
                    candidate_set = find_blocks_covering_tsub_list(combined)
                    if candidate_set:
                        candidate_set.discard(b1)
                        candidate_set.discard(b2)
                        if candidate_set:
                            # Check if this swap is beneficial (reduces total unique coverage loss)
                            best_new = None
                            best_score = -1
                            for b_new_idx in list(candidate_set)[:20]:
                                score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                                if score > best_score:
                                    best_score = score
                                    best_new = b_new_idx

                            if best_new is not None:
                                remove_block(b1)
                                remove_block(b2)
                                add_block(best_new)
                                improved = True
                                pair_swapped = True
                                break

                if pair_swapped:
                    break

            if pair_swapped:
                no_improvement_count = 0
                continue

        # Phase 4: Try 1-for-2 swaps (remove 1 block, add 2)
        if time.time() < deadline - 5:
            all_critical = [b for b in blocks_in_cover if np.any(cover_count[block_to_tsubs_np[b]] == 1)]
            random.shuffle(all_critical)

            triple_swapped = False
            for b_old_idx in all_critical:
                if time.time() > deadline - 5:
                    break

                unique_tsubs = get_unique_tsubs_for_block(b_old_idx)
                if len(unique_tsubs) == 0:
                    continue

                # We need to cover these unique t-subsets with 2 new blocks
                # Try each t-subset in unique_tsubs and find blocks covering it
                # Then find a second block that covers the remaining unique t-subsets
                best_pair = None
                best_pair_score = -1

                for t_idx in unique_tsubs[:5]:  # Limit for speed
                    blocks_covering_t = list(t_sub_to_blocks[int(t_idx)])
                    blocks_covering_t = [b for b in blocks_covering_t if b != b_old_idx]

                    for b1_new in blocks_covering_t[:10]:
                        # What unique t-subsets does b1_new NOT cover?
                        b1_tsubs = set(block_to_tsubs_np[b1_new].tolist())
                        remaining = [t for t in unique_tsubs if int(t) not in b1_tsubs]

                        if not remaining:
                            # b1_new covers all unique t-subsets, so we just need one more block
                            # This is actually a 1-for-1 swap, which we already tried
                            continue

                        # Find a block that covers all remaining unique t-subsets
                        candidate_set2 = find_blocks_covering_tsub_list(remaining)
                        if candidate_set2:
                            candidate_set2.discard(b_old_idx)
                            candidate_set2.discard(b1_new)
                            if candidate_set2:
                                for b2_new in list(candidate_set2)[:10]:
                                    score = (np.sum(cover_count[block_to_tsubs_np[b1_new]] == 0) +
                                             np.sum(cover_count[block_to_tsubs_np[b2_new]] == 0))
                                    if score > best_pair_score:
                                        best_pair_score = score
                                        best_pair = (b1_new, b2_new)

                if best_pair is not None:
                    remove_block(b_old_idx)
                    add_block(best_pair[0])
                    add_block(best_pair[1])
                    improved = True
                    triple_swapped = True
                    break

            if triple_swapped:
                no_improvement_count = 0
                continue

        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            break

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True
>>>>>>> REPLACE

