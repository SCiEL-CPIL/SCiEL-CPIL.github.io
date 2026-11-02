    def local_search_improve(selected_indices, conflict_matrix, n, time_limit, max_iter=200):
        """
        Improve the set by trying to add vertices or swap.
        """
        selected = set(selected_indices)
        sel_mask = np.zeros(n, dtype=bool)
        for idx in selected:
            sel_mask[idx] = True
            
        improved = True
        stall_count = 0
        while improved and time.time() < time_limit and stall_count < 10:
            improved = False
            stall_count += 1
            
            # 1. Try to add
            conflict_counts = conflict_matrix @ sel_mask
            addable = (~sel_mask) & (conflict_counts == 0)
            
            if addable.any():
                candidates_to_add = np.where(addable)[0]
                # Heuristic: pick the one with minimum degree in the conflict graph
                # deg = conflict_matrix @ np.ones(n)
                # Precompute degrees? 
                # For now, pick first.
                new_idx = candidates_to_add[0]
                
                selected.add(new_idx)
                sel_mask[new_idx] = True
                improved = True
                stall_count = 0
                continue
            
            # 2. Try to swap
            # Pick a random selected vertex to remove
            if len(selected) > 0:
                s = np.random.choice(list(selected))
                # Remove s
                selected.remove(s)
                sel_mask[s] = False
                
                # Try to add any vertex compatible with the new set
                conflict_counts = conflict_matrix @ sel_mask
                addable = (~sel_mask) & (conflict_counts == 0)
                
                if addable.any():
                    candidates_to_add = np.where(addable)[0]
                    # Prefer one that conflicts with s? No, s is removed.
                    # Just pick one.
                    new_idx = candidates_to_add[0]
                    selected.add(new_idx)
                    sel_mask[new_idx] = True
                    improved = True
                    stall_count = 0
                    continue
                else:
                    # Put s back if we couldn't replace it effectively?
                    # Or keep it removed? 
                    # If we can't add anything, removing s just makes the set smaller.
                    # So we should put s back.
                    selected.add(s)
                    sel_mask[s] = True
                    # improved remains False
            else:
                break
                
        return list(selected)
