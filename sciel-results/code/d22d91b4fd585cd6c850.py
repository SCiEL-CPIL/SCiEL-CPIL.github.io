<<<<<<< SEARCH
    # 6. ILP Optimization
    if time.monotonic() < deadline - 10.0:
        try:
            remaining_time = deadline - time.monotonic() - 5.0
            if remaining_time > 5.0:
                c = np.ones(num_kblocks)
                constraints = LinearConstraint(A, 1, np.inf)
                integrality = np.ones(num_kblocks)
                bounds = Bounds(0, 1)
                
                res = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    time_limit=min(remaining_time, 30.0)
                )
                
                if res.x is not None:
                    x_sol = np.round(res.x)
                    if np.all((x_sol == 0) | (x_sol == 1)):
                        coverage = A @ x_sol
                        if np.all(coverage >= 1):
                            new_count = np.sum(x_sol)
                            old_count = len(selected_blocks)
                            if new_count < old_count:
                                selected_blocks = [i for i in range(num_kblocks) if x_sol[i] == 1]
                                
        except Exception:
            pass
=======
    # 6. ILP Optimization
    if time.monotonic() < deadline - 10.0:
        try:
            remaining_time = deadline - time.monotonic() - 5.0
            if remaining_time > 5.0:
                c = np.ones(num_kblocks)
                constraints = LinearConstraint(A, 1, np.inf)
                integrality = np.ones(num_kblocks)
                bounds = Bounds(0, 1)
                
                res = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    time_limit=min(remaining_time, 40.0)
                )
                
                if res.x is not None:
                    x_sol = np.round(res.x)
                    if np.all((x_sol == 0) | (x_sol == 1)):
                        coverage = A @ x_sol
                        if np.all(coverage >= 1):
                            new_count = int(np.sum(x_sol))
                            old_count = len(selected_blocks)
                            if new_count < old_count:
                                selected_blocks = [i for i in range(num_kblocks) if x_sol[i] == 1]
                                # Polish the ILP result
                                counts2 = np.zeros(num_tsubs, dtype=int)
                                for b_idx in selected_blocks:
                                    for idx in block_tsub_idxs[b_idx]:
                                        counts2[idx] += 1
                                changed2 = True
                                while changed2 and time.monotonic() < deadline - 5.0:
                                    changed2 = False
                                    for i2 in range(len(selected_blocks)):
                                        b_idx2 = selected_blocks[i2]
                                        can_remove2 = True
                                        for idx in block_tsub_idxs[b_idx2]:
                                            if counts2[idx] == 1:
                                                can_remove2 = False
                                                break
                                        if can_remove2:
                                            selected_blocks.pop(i2)
                                            for idx in block_tsub_idxs[b_idx2]:
                                                counts2[idx] -= 1
                                            changed2 = True
                                            break
                                
        except Exception:
            pass
>>>>>>> REPLACE
