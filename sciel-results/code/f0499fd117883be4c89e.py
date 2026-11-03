<<<<<<< SEARCH
    # Run multiple greedy restarts with different weight modes
    best_selected = None
    best_len = float('inf')

    # Restart 1: standard inverse-count weights
    sel1, _ = run_greedy('inverse', 42)
    sel1 = local_search(sel1, None)
    if len(sel1) < best_len:
        best_len = len(sel1)
        best_selected = sel1[:]

    if time_left() > 40:
        # Restart 2: perturbed weights
        sel2, _ = run_greedy('random', 123)
        sel2 = local_search(sel2, None)
        if len(sel2) < best_len:
            best_len = len(sel2)
            best_selected = sel2[:]

    if time_left() > 40:
        # Restart 3: different perturbation
        sel3, _ = run_greedy('random', 777)
        sel3 = local_search(sel3, None)
        if len(sel3) < best_len:
            best_len = len(sel3)
            best_selected = sel3[:]

    # Final local search on best
    if best_selected and time_left() > 10:
        best_selected = local_search(best_selected, None)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]

    return result
=======
    # Run multiple greedy restarts with different weight modes
    best_selected = None
    best_len = float('inf')

    # Restart 1: standard inverse-count weights
    sel1, _ = run_greedy('inverse', 42)
    sel1 = local_search(sel1, None)
    if len(sel1) < best_len:
        best_len = len(sel1)
        best_selected = sel1[:]

    if time_left() > 35:
        # Restart 2: perturbed weights
        sel2, _ = run_greedy('random', 123)
        sel2 = local_search(sel2, None)
        if len(sel2) < best_len:
            best_len = len(sel2)
            best_selected = sel2[:]

    if time_left() > 35:
        # Restart 3: different perturbation
        sel3, _ = run_greedy('random2', 777)
        sel3 = local_search(sel3, None)
        if len(sel3) < best_len:
            best_len = len(sel3)
            best_selected = sel3[:]

    if time_left() > 35:
        # Restart 4: another perturbation
        sel4, _ = run_greedy('random', 2024)
        sel4 = local_search(sel4, None)
        if len(sel4) < best_len:
            best_len = len(sel4)
            best_selected = sel4[:]

    if time_left() > 35:
        # Restart 5: yet another
        sel5, _ = run_greedy('random2', 999)
        sel5 = local_search(sel5, None)
        if len(sel5) < best_len:
            best_len = len(sel5)
            best_selected = sel5[:]

    # Final local search on best
    if best_selected and time_left() > 8:
        best_selected = local_search(best_selected, None)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]

    return result
>>>>>>> REPLACE
