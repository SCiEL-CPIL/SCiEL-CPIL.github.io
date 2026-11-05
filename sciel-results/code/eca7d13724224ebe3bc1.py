<<<<<<< SEARCH
    # Iterative pair-replacement: remove two blocks, add one covering all lost triples
    improved = True
    while improved:
        improved = False
        for i in range(len(sol)):
            for j in range(i + 1, len(sol)):
                b1 = sol[i]
                b2 = sol[j]
                tris1 = block_triple_arrays[b1]
                tris2 = block_triple_arrays[b2]
                if len(tris1) == 0 or len(tris2) == 0:
                    continue
                sim_counts = counts.copy()
                sim_counts[tris1] -= 1
                sim_counts[tris2] -= 1
                uncovered_tris = np.where(sim_counts == 0)[0]
                if len(uncovered_tris) == 0:
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    sol.remove(b1)
                    sol.remove(b2)
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    improved = True
                    break
                elif len(uncovered_tris) <= 5:
                    candidate_blocks = set()
                    for t_idx in uncovered_tris:
                        candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                    candidate_blocks.difference_update(sol_set)
                    found = False
                    if candidate_blocks:
                        for cand in candidate_blocks:
                            tris_cand = block_triple_arrays[cand]
                            if np.all(np.isin(uncovered_tris, tris_cand)):
                                counts[tris1] -= 1
                                counts[tris2] -= 1
                                counts[tris_cand] += 1
                                sol.remove(b1)
                                sol.remove(b2)
                                sol_set.discard(b1)
                                sol_set.discard(b2)
                                sol.append(cand)
                                sol_set.add(cand)
                                improved = True
                                found = True
                                break
                    if found:
                        break
        # Recompute counts after structural changes to stay consistent
        if improved:
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in sol:
                tris = block_triple_arrays[idx]
                if len(tris) > 0:
                    counts[tris] += 1
            sol_set = set(sol)

    return sol
=======
    # Iterative pair-replacement: remove two blocks, add one covering all lost triples
    improved = True
    while improved:
        improved = False
        for i in range(len(sol)):
            for j in range(i + 1, len(sol)):
                b1 = sol[i]
                b2 = sol[j]
                tris1 = block_triple_arrays[b1]
                tris2 = block_triple_arrays[b2]
                if len(tris1) == 0 or len(tris2) == 0:
                    continue
                sim_counts = counts.copy()
                sim_counts[tris1] -= 1
                sim_counts[tris2] -= 1
                uncovered_tris = np.where(sim_counts == 0)[0]
                if len(uncovered_tris) == 0:
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    sol.remove(b1)
                    sol.remove(b2)
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    improved = True
                    break
                elif len(uncovered_tris) <= 5:
                    candidate_blocks = set()
                    for t_idx in uncovered_tris:
                        candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                    candidate_blocks.difference_update(sol_set)
                    found = False
                    if candidate_blocks:
                        for cand in candidate_blocks:
                            tris_cand = block_triple_arrays[cand]
                            if np.all(np.isin(uncovered_tris, tris_cand)):
                                counts[tris1] -= 1
                                counts[tris2] -= 1
                                counts[tris_cand] += 1
                                sol.remove(b1)
                                sol.remove(b2)
                                sol_set.discard(b1)
                                sol_set.discard(b2)
                                sol.append(cand)
                                sol_set.add(cand)
                                improved = True
                                found = True
                                break
                    if found:
                        break
        # Recompute counts after structural changes to stay consistent
        if improved:
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in sol:
                tris = block_triple_arrays[idx]
                if len(tris) > 0:
                    counts[tris] += 1
            sol_set = set(sol)

    # Final single-block replacement: try replacing one block with another that covers same + more
    improved = True
    while improved:
        improved = False
        for idx in list(sol):
            tris_idx = block_triple_arrays[idx]
            if len(tris_idx) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                improved = True
                continue
            # Check if this block is redundant (all its triples covered by others)
            if np.all(counts[tris_idx] > 1):
                counts[tris_idx] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                improved = True
                continue
            # Try to find a block that covers all triples of idx and at least one more
            lost_tris = tris_idx[counts[tris_idx] == 1]
            if len(lost_tris) == 0:
                counts[tris_idx] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                improved = True
                continue
            # Find candidates that cover all lost_tris
            candidate_blocks = set()
            for t_idx in lost_tris:
                candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
            candidate_blocks.difference_update(sol_set)
            candidate_blocks.discard(idx)
            for cand in candidate_blocks:
                tris_cand = block_triple_arrays[cand]
                if np.all(np.isin(lost_tris, tris_cand)):
                    # Check if cand adds new coverage
                    new_tris = tris_cand[~np.isin(tris_cand, tris_idx)]
                    if len(new_tris) > 0:
                        counts[tris_idx] -= 1
                        counts[tris_cand] += 1
                        sol.remove(idx)
                        sol_set.discard(idx)
                        sol.append(cand)
                        sol_set.add(cand)
                        improved = True
                        break
        if improved:
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in sol:
                tris = block_triple_arrays[idx]
                if len(tris) > 0:
                    counts[tris] += 1
            sol_set = set(sol)

    return sol
>>>>>>> REPLACE
