<<<<<<< SEARCH
    # Local search with bitset-based can_add
    # We'll use a temperature-based approach for more diverse exploration
    
    while time.time() < deadline:
        improved = False
        S_list = sorted(S)
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add_bitset(bitset, x):
                new_S = S | {x}
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S = new_S
                    bitset[x] = True
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where v < u)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                S_temp = S - {u}
                bitset_temp = bitset.copy()
                bitset_temp[u] = False
                temp_val = best_val - 1.0 / u
                
                for v in excluded:
                    if v == u:
                        continue
                    if v >= u:
                        break  # We want v < u for improvement
                    if can_add_bitset(bitset_temp, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S = S_temp | {v}
                            bitset = bitset_temp.copy()
                            bitset[v] = True
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            break  # Restart swap search with new S
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            # Only do this if we have time
            if time.time() > deadline - 5.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            # Try removing the two largest elements
            if len(S_list) >= 2:
                u1, u2 = S_list[0], S_list[1]
                S_temp = S - {u1} - {u2}
                bitset_temp = bitset.copy()
                bitset_temp[u1] = False
                bitset_temp[u2] = False
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded[:20]):  # Only try top 20 candidates
                    if v1 == u1 or v1 == u2:
                        continue
                    if can_add_bitset(bitset_temp, v1):
                        bitset_temp2 = bitset_temp.copy()
                        bitset_temp2[v1] = True
                        for v2 in excluded[i+1:20]:
                            if v2 == u1 or v2 == u2:
                                continue
                            if can_add_bitset(bitset_temp2, v2):
                                new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                if new_val > best_val + 1e-15:
                                    S = S_temp | {v1, v2}
                                    bitset = bitset_temp2.copy()
                                    bitset[v2] = True
                                    best_S = set(S)
                                    best_val = new_val
                                    found_improvement = True
                                    break
                    if found_improvement:
                        break
            
            if found_improvement:
                continue  # Loop back to try more improvements
            else:
                break  # No improvement possible
    
    return list(best_S)
=======
    # Local search with fast can_add
    while time.time() < deadline:
        improved = False
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x for max reciprocal)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add_fast(membership, elements_list, x):
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S.add(x)
                    membership[x] = True
                    elements_list.append(x)
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where v < u for improvement)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                # Temporarily remove u
                membership[u] = False
                elements_list.remove(u)
                temp_val = best_val - 1.0 / u
                
                found_swap = False
                for v in excluded_list:
                    if v == u:
                        continue
                    if v >= u:
                        break  # We want v < u for improvement
                    if can_add_fast(membership, elements_list, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S.remove(u)
                            S.add(v)
                            membership[u] = False
                            membership[v] = True
                            elements_list.append(v)
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            found_swap = True
                            break  # Restart swap search with new S
                
                if not found_swap:
                    # Restore u
                    membership[u] = True
                    elements_list.append(u)
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            if time.time() > deadline - 5.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            if len(S_list) >= 2:
                u1, u2 = S_list[0], S_list[1]
                # Temporarily remove u1, u2
                membership[u1] = False
                membership[u2] = False
                elements_list.remove(u1)
                elements_list.remove(u2)
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded_list[:30]):
                    if v1 == u1 or v1 == u2:
                        continue
                    if can_add_fast(membership, elements_list, v1):
                        membership[v1] = True
                        elements_list.append(v1)
                        for v2 in excluded_list[i+1:30]:
                            if v2 == u1 or v2 == u2 or v2 == v1:
                                continue
                            if can_add_fast(membership, elements_list, v2):
                                new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                if new_val > best_val + 1e-15:
                                    S.discard(u1)
                                    S.discard(u2)
                                    S.add(v1)
                                    S.add(v2)
                                    membership[v1] = True
                                    membership[v2] = True
                                    elements_list.append(v2)
                                    best_S = set(S)
                                    best_val = new_val
                                    found_improvement = True
                                    break
                        # Restore v1 if no improvement found with it
                        if not found_improvement:
                            membership[v1] = False
                            elements_list.remove(v1)
                    if found_improvement:
                        break
                
                if not found_improvement:
                    # Restore u1, u2
                    membership[u1] = True
                    membership[u2] = True
                    elements_list.append(u1)
                    elements_list.append(u2)
            
            if found_improvement:
                continue
            else:
                break
    
    return list(best_S)
>>>>>>> REPLACE

