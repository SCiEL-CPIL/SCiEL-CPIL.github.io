<<<<<<< SEARCH
    # Multiple restarts with different seeds and strategies
    attempts = 0
    strategies = ['random', 'bit_priority', 'parity_weighted', 'clique_breaking', 'random', 'random']
    
    while time_left() > 15:
        attempts += 1
        seed = attempts * 7919 + 42
        strategy = strategies[attempts % len(strategies)]
        
        # Alternate between random greedy and perturbed deterministic
        if attempts % 5 == 1:
            sol = greedy_initial(seed, strategy)
        elif attempts % 5 == 2:
            sol = list(best_solution) if best_solution else greedy_initial(seed, strategy)
            # Perturb: remove a few random edges
            rng = random.Random(seed)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(5, len(selected) // 10))
                for ei in to_remove:
                    sol[ei] = False
        elif attempts % 5 == 3:
            sol = greedy_initial(seed + 12345, 'random')
        elif attempts % 5 == 4:
            sol = greedy_initial(seed + 99999, 'bit_priority')
        else:
            sol = greedy_initial(seed + 55555, 'parity_weighted')
        
        # Local search
        sol = local_search(sol, seed)
        
        # Additional swap search
        if time_left() > 30:
            sol = swap_search(sol, seed + 777)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 30:
            break
=======
    # Multiple restarts with different seeds and strategies
    attempts = 0
    strategies = ['random', 'bit_priority', 'parity_weighted', 'random', 'random', 'bit_priority']
    
    while time_left() > 25:
        attempts += 1
        seed = attempts * 7919 + 42
        strategy = strategies[attempts % len(strategies)]
        
        if attempts % 5 == 1:
            sol = greedy_initial(seed, strategy)
        elif attempts % 5 == 2:
            sol = list(best_solution) if best_solution else greedy_initial(seed, strategy)
            rng = random.Random(seed)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(5, len(selected) // 10))
                for ei in to_remove:
                    sol[ei] = False
        elif attempts % 5 == 3:
            sol = greedy_initial(seed + 12345, 'random')
        elif attempts % 5 == 4:
            sol = greedy_initial(seed + 99999, 'bit_priority')
        else:
            sol = greedy_initial(seed + 55555, 'parity_weighted')
        
        sol = local_search(sol, seed)
        
        if time_left() > 40:
            sol = swap_search(sol, seed + 777)
        
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
    
    # Final polish: try to add any remaining edges and do targeted swaps
    if best_solution and time_left() > 12:
        sol = list(best_solution)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        # Add any edge that fits
        for ei in range(n_edges):
            if sol[ei]:
                continue
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] >= 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        # Targeted swaps for missing edges
        missing = [ei for ei in range(n_edges) if not sol[ei]]
        rng = random.Random(99999)
        rng.shuffle(missing)
        for ei_add in missing:
            if time_left() < 12:
                break
            blocking = [fi for fi in edge_faces[ei_add] if face_counts[fi] >= 3]
            if not blocking:
                continue
            for fi in blocking:
                for ei_rem in faces[fi]:
                    if not sol[ei_rem] or ei_rem == ei_add:
                        continue
                    temp_fc = face_counts.copy()
                    for f2 in edge_faces[ei_rem]:
                        temp_fc[f2] -= 1
                    valid = True
                    for f2 in edge_faces[ei_add]:
                        if temp_fc[f2] >= 3:
                            valid = False
                            break
                    if valid:
                        sol[ei_rem] = False
                        sol[ei_add] = True
                        for f2 in edge_faces[ei_rem]:
                            face_counts[f2] -= 1
                        for f2 in edge_faces[ei_add]:
                            face_counts[f2] += 1
                        break
                if time_left() < 12:
                    break
        cnt = count_edges(sol)
        if is_valid(sol) and cnt > best_count:
            best_count = cnt
            best_solution = list(sol)
>>>>>>> REPLACE
