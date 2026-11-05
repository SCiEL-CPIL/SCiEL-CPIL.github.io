<<<<<<< SEARCH
    # Step 1: Generate candidate pool
    candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=400)

    # Deduplicate by difference set
    seen_diffs = set()
    unique_candidates = []
    for row, diffs in candidates:
        if diffs not in seen_diffs:
            seen_diffs.add(diffs)
            unique_candidates.append((row, diffs))

    candidates = unique_candidates

    # Step 2: Find maximum clique if time permits
    if len(candidates) >= n_rows and time.time() < deadline - 60:
        adj = _build_compatibility_graph(candidates)
        clique = _max_clique_branch_and_bound(adj, deadline - 30)

        if len(clique) >= n_rows:
            selected = clique[:n_rows]
            rows = [candidates[i][0] for i in selected]
        else:
            rows = _greedy_construct(n_rows, k)
            if rows is None:
                rows = []
                used = set()
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                    rows.append(row)
                return rows
    else:
        rows = _greedy_construct(n_rows, k)
        if rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    # Step 3: Local search to reduce scope
    if time.time() < deadline - 15:
        rows = _local_search_scope(rows, k, rng, deadline - 15)

    # Step 4: Additional local search iterations
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            rows = _local_search_scope(rows, k, rng, deadline - 10)

    # Step 5: Polish pass
    if time.time() < deadline - 3:
        rows = _polish_pass(rows, k, deadline - 3)

    return rows
=======
    # Step 1: Generate initial solution via greedy
    initial_rows = _greedy_construct(n_rows, k)
    if initial_rows is None:
        # Fallback: pure greedy without any optimization
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Step 2: Run simulated annealing with multiple restarts
    best_rows = initial_rows
    best_scope = _total_scope(best_rows)
    
    # Allocate time for SA: ~60% of budget
    sa_deadline = start_time + 80
    
    for restart in range(5):
        if time.time() > sa_deadline:
            break
            
        # Generate diverse initial solution for this restart
        if restart == 0:
            sa_initial = initial_rows
        else:
            # Perturb the best solution found so far
            sa_initial = [r[:] for r in best_rows]
            # Randomly modify a few elements
            for _ in rng.randint(1, 3):
                idx = rng.randrange(len(sa_initial))
                pos = rng.randrange(1, len(sa_initial[idx]))
                sa_initial[idx][pos] += rng.randint(-3, 3)
                # Ensure validity of this row
                if not all(sa_initial[idx][i] < sa_initial[idx][i+1] for i in range(len(sa_initial[idx])-1)):
                    sa_initial[idx][pos] = sa_initial[idx][pos-1] + 1
                    for j in range(pos+1, len(sa_initial[idx])):
                        sa_initial[idx][j] = sa_initial[idx][j-1] + 1

        sa_rows = _simulated_annealing(n_rows, k, rng, sa_deadline, initial_rows=sa_initial)
        if sa_rows is not None and _is_valid_set(sa_rows):
            sa_scope = _total_scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

    # Step 3: Local search to reduce scope
    if time.time() < deadline - 15:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 15)

    # Step 4: Additional local search iterations
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            best_rows = _local_search_scope(best_rows, k, rng, deadline - 10)

    # Step 5: Polish pass
    if time.time() < deadline - 3:
        best_rows = _polish_pass(best_rows, k, deadline - 3)

    return best_rows
>>>>>>> REPLACE
