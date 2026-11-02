    def local_search(sol_indices):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in sol_indices:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(sol_indices)):
                    if time.time() > deadline - 5.0:
                        return sol_indices, cover_count
                    b_idx = sol_indices[i]
                    redundant = True
                    for ts_idx in block_to_ts[b_idx]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ts_idx in block_to_ts[b_idx]:
                            cover_count[ts_idx] -= 1
                        sol_indices.pop(i)
                        changed = True
                        improved = True
                        break
                        
            if not improved and time.time() > deadline - 5.0:
                return sol_indices, cover_count

            # 2. Try 2-for-1 merge
            n = len(sol_indices)
            found_merge = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    return sol_indices, cover_count
                b_i = sol_indices[i]
                unique_i = [ts for ts in block_to_ts[b_i] if cover_count[ts] == 1]
                
                for j in range(i + 1, n):
                    if time.time() > deadline - 5.0:
                        return sol_indices, cover_count
                    b_j = sol_indices[j]
                    unique_j = [ts for ts in block_to_ts[b_j] if cover_count[ts] == 1]
                    
                    all_unique = unique_i + unique_j
                    if not all_unique:
                        continue
                        
                    sorted_uniques = sorted(all_unique, key=lambda x: len(ts_to_blocks[x]))
                    candidates = set(ts_to_blocks[sorted_uniques[0]])
                    for ts_idx in sorted_uniques[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if candidates:
                        sol_set = set(sol_indices)
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        if valid_candidates:
                            new_b_idx = random.choice(valid_candidates)
                            
                            for ts_idx in block_to_ts[b_i]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[b_j]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[new_b_idx]:
                                cover_count[ts_idx] += 1
                                
                            sol_indices.pop(j)
                            sol_indices.pop(i)
                            sol_indices.append(new_b_idx)
                            
                            found_merge = True
                            improved = True
                            break
                if found_merge:
                    break
                    
            # 3. Try 1-for-1 swap
            if not improved:
                n = len(sol_indices)
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        return sol_indices, cover_count
                    b_i = sol_indices[i]
                    unique_i = [ts for ts in block_to_ts[b_i] if cover_count[ts] == 1]
                    if not unique_i:
                        continue
                        
                    sorted_uniques = sorted(unique_i, key=lambda x: len(ts_to_blocks[x]))
                    candidates = set(ts_to_blocks[sorted_uniques[0]])
                    for ts_idx in sorted_uniques[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if candidates:
                        sol_set = set(sol_indices)
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        if valid_candidates:
                            new_b_idx = random.choice(valid_candidates)
                            
                            for ts_idx in block_to_ts[b_i]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[new_b_idx]:
                                cover_count[ts_idx] += 1
                                
                            sol_indices[i] = new_b_idx
                            improved = True
                            break

        return sol_indices, cover_count
