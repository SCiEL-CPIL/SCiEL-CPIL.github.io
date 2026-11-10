<<<<<<< SEARCH
        # If no 1-add, try 1-swaps
        # We need to be careful: a 1-swap might allow a future 1-add.
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) == 0 or len(non_edges) == 0:
            break
            
        # Try a limited number of 1-swaps
        swaps_tried = 0
        max_swaps_to_try = 50
        found_1swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            rem_u, rem_v = random.choice(cur_edges)
            add_u, add_v = random.choice(non_edges)
            
            # Ensure they are distinct
            if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                continue
                
            # Apply removal
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            
            # Apply addition
            adj[add_u] |= (1 << add_v)
            adj[add_v] |= (1 << add_u)
            
            # Check if C6-free
            # Only need to check the added edge? 
            # Removing edges cannot create C6. Adding edge (add_u, add_v) might.
            if _check_c6_through_edge(adj, add_u, add_v, n):
                # Valid swap
                found_1swap = True
                break
            else:
                # Revert
                adj[add_u] &= ~(1 << add_v)
                adj[add_v] &= ~(1 << add_u)
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                
        if found_1swap:
            # We made a 1-swap. This doesn't increase edge count, but might open up for 1-add.
            # We don't update best_count, but we continue the loop to try 1-add again.
            continue
        
        # If no 1-swap found (or tried enough), try 2-swaps
        # Reset lists as they might be stale or we want fresh randomness
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
            
        swaps_tried = 0
        max_swaps_to_try = 100
        found_2swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            # Pick 2 distinct edges to remove
            e1, e2 = random.sample(cur_edges, 2)
            # Pick 2 distinct non-edges to add
            a1, a2 = random.sample(non_edges, 2)
            
            # Check for overlaps (shouldn't happen as one is edge, one is non-edge, but just in case)
            # Actually, e1, e2 are edges, a1, a2 are non-edges. So no overlap.
            
            # Apply removals
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            
            # Apply additions
            adj[a1[0]] |= (1 << a1[1])
            adj[a1[1]] |= (1 << a1[0])
            adj[a2[0]] |= (1 << a2[1])
            adj[a2[1]] |= (1 << a2[0])
            
            # Check if C6-free
            is_valid = True
            if not _check_c6_through_edge(adj, a1[0], a1[1], n):
                is_valid = False
            elif not _check_c6_through_edge(adj, a2[0], a2[1], n):
                is_valid = False
            
            if is_valid:
                found_2swap = True
                break
            else:
                # Revert
                adj[a2[0]] &= ~(1 << a2[1])
                adj[a2[1]] &= ~(1 << a2[0])
                adj[a1[0]] &= ~(1 << a1[1])
                adj[a1[1]] &= ~(1 << a1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                
        if found_2swap:
            continue
            
        # If no moves found, we are stuck.
        # To avoid infinite loop, we can break or try a larger perturbation.
        # For now, break out of this local search instance.
        break
=======
        # If no 1-add, try 1-swaps
        # We need to be careful: a 1-swap might allow a future 1-add.
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) == 0 or len(non_edges) == 0:
            break
            
        # Try a limited number of 1-swaps
        swaps_tried = 0
        max_swaps_to_try = 80
        found_1swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            rem_u, rem_v = random.choice(cur_edges)
            add_u, add_v = random.choice(non_edges)
            
            # Ensure they are distinct
            if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                continue
                
            # Apply removal
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            
            # Apply addition
            adj[add_u] |= (1 << add_v)
            adj[add_v] |= (1 << add_u)
            
            # Check if C6-free
            # Only need to check the added edge? 
            # Removing edges cannot create C6. Adding edge (add_u, add_v) might.
            if _check_c6_through_edge(adj, add_u, add_v, n):
                # Valid swap
                found_1swap = True
                break
            else:
                # Revert
                adj[add_u] &= ~(1 << add_v)
                adj[add_v] &= ~(1 << add_u)
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                
        if found_1swap:
            # We made a 1-swap. This doesn't increase edge count, but might open up for 1-add.
            # We don't update best_count, but we continue the loop to try 1-add again.
            continue
        
        # If no 1-swap found (or tried enough), try 2-swaps
        # Reset lists as they might be stale or we want fresh randomness
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
            
        swaps_tried = 0
        max_swaps_to_try = 150
        found_2swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            # Pick 2 distinct edges to remove
            e1, e2 = random.sample(cur_edges, 2)
            # Pick 2 distinct non-edges to add
            a1, a2 = random.sample(non_edges, 2)
            
            # Apply removals
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            
            # Apply additions
            adj[a1[0]] |= (1 << a1[1])
            adj[a1[1]] |= (1 << a1[0])
            adj[a2[0]] |= (1 << a2[1])
            adj[a2[1]] |= (1 << a2[0])
            
            # Check if C6-free
            is_valid = True
            if not _check_c6_through_edge(adj, a1[0], a1[1], n):
                is_valid = False
            elif not _check_c6_through_edge(adj, a2[0], a2[1], n):
                is_valid = False
            
            if is_valid:
                found_2swap = True
                break
            else:
                # Revert
                adj[a2[0]] &= ~(1 << a2[1])
                adj[a2[1]] &= ~(1 << a2[0])
                adj[a1[0]] &= ~(1 << a1[1])
                adj[a1[1]] &= ~(1 << a1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                
        if found_2swap:
            continue
            
        # If no moves found, we are stuck.
        # Try a perturbation: randomly remove 3 edges and re-add 3
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 3 or len(non_edges) < 3:
            break
            
        # Perturbation: remove 3 random edges, add 3 random non-edges
        rem_edges = random.sample(cur_edges, 3)
        add_edges = random.sample(non_edges, 3)
        
        for ru, rv in rem_edges:
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
        
        valid = True
        for au, av in add_edges:
            adj[au] |= (1 << av)
            adj[av] |= (1 << au)
            if not _check_c6_through_edge(adj, au, av, n):
                valid = False
                break
        
        if not valid:
            # Revert
            for au, av in add_edges:
                adj[au] &= ~(1 << av)
                adj[av] &= ~(1 << au)
            for ru, rv in rem_edges:
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
            break
        else:
            # Accept perturbation, continue searching
            continue
>>>>>>> REPLACE
