<<<<<<< SEARCH
def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # Increase attempts for larger n where finding valid graphs is harder
        if n >= 21:
            max_attempts = 12
        elif n >= 19:
            max_attempts = 10
        else:
            max_attempts = 15

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability: try a range of densities
            # Regular induced subgraphs are more likely at p=0.5, so avoid that
            p_choices = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            p = p_choices[rng.randrange(len(p_choices))]

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue

    if not best_valid or best_n < 5:
        # Fallback: return a small valid graph
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
=======
def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # Increase attempts for larger n where finding valid graphs is harder
        if n >= 21:
            max_attempts = 10
        elif n >= 19:
            max_attempts = 8
        else:
            max_attempts = 12

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability: try a range of densities
            # Regular induced subgraphs are more likely at p=0.5, so avoid that
            p_choices = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            p = p_choices[rng.randrange(len(p_choices))]

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue

    if not best_valid or best_n < 5:
        # Fallback: return a small valid graph
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
>>>>>>> REPLACE
