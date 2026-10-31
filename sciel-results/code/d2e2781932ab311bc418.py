                idx_remove = np.random.choice(selected)
                sel_set.remove(idx_remove)
                
                found = False
                for _ in range(100):
                    cand = np.random.randint(n)
                    if cand in sel_set:
                        continue
                    
                    # Check if cand conflicts with any in sel_set
                    # sel_set is now selected - {idx_remove}
                    is_conflict = False
                    for nb in neighbors[cand]:
                        if nb in sel_set:
                            is_conflict = True
                            break
                    
                    if not is_conflict:
                        # Valid candidate
                        selected.remove(idx_remove)
                        selected.append(cand)
                        sel_set.add(cand)
                        improved = True
                        swaps_done += 1
                        found = True
                        break
                
                if not found:
                    # Restore
                    sel_set.add(idx_remove)
