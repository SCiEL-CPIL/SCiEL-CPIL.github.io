<<<<<<< SEARCH
    best_solution = set()

    # 1. Exact/branch-and-bound on the full candidate pool (fast due to matching structure).
    sol = maximize(candidates, deadline - 10.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    improved = list(best_solution)
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            improved.append(a)
            best_solution.add(a)

    # 3. Local search with add/remove/swap moves (inspiration 1 style, more aggressive)
    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

        return s

    best_solution = local_search_improve(best_solution, deadline - 10.0)

    # 4. Multiple random restarts of greedy + local search if time permits
    if not time_up(deadline - 15.0):
        for restart in range(50):
            if time_up(deadline - 5.0):
                break
            # Randomized greedy
            rem = set(candidates)
            greedy_set = set()
            while rem:
                # Pick random low-degree node
                low_deg_nodes = []
                min_d = float('inf')
                for node in rem:
                    d = len(conflict[node] & rem)
                    if d < min_d:
                        min_d = d
                        low_deg_nodes = [node]
                    elif d == min_d:
                        low_deg_nodes.append(node)
                chosen = random.choice(low_deg_nodes)
                greedy_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
            # Local search on this greedy result
            ls_result = local_search_improve(greedy_set, deadline - 5.0)
            if len(ls_result) > len(best_solution):
                best_solution = ls_result
=======
    best_solution = set()

    # 1. Exact/branch-and-bound on the full candidate pool (fast due to matching structure).
    sol = maximize(candidates, deadline - 10.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            best_solution.add(a)

    # 3. Enhanced local search with add/swap/2-for-1/2-for-2 moves
    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try 2-for-1 moves: remove u, add two non-conflicting nodes
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                # Check if any two addable nodes are non-conflicting
                added = False
                for i in range(len(addable)):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, len(addable)):
                        if not (conflict[addable[i]] & {addable[j]}) and not (conflict[addable[j]] & {addable[i]}):
                            s.add(addable[i])
                            s.add(addable[j])
                            added = True
                            break
                    if added:
                        break
                if added:
                    improved_flag = True
                    break
                else:
                    # Try 1-for-1 swap if beneficial
                    if addable:
                        best_v = None
                        best_deg = float('inf')
                        for v in addable:
                            d = len(conflict[v] & s)
                            if d < best_deg:
                                best_deg = d
                                best_v = v
                        deg_u = len(conflict[u] & s)
                        if best_v is not None and best_deg < deg_u:
                            s.add(best_v)
                            improved_flag = True
                            break
                    s.add(u)

            # 3. Try 2-for-2 moves: remove two nodes, add two different nodes
            if not improved_flag:
                s_list = list(s)
                for i in range(len(s_list)):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, len(s_list)):
                        if time_up(time_budget):
                            break
                        u1, u2 = s_list[i], s_list[j]
                        s.remove(u1)
                        s.remove(u2)
                        
                        addable = []
                        for v in candidates:
                            if v not in s and v != u1 and v != u2:
                                if try_add_fast(s, v):
                                    addable.append(v)
                        
                        added = False
                        for a in range(len(addable)):
                            if time_up(time_budget):
                                break
                            for b in range(a + 1, len(addable)):
                                if not (conflict[addable[a]] & {addable[b]}) and not (conflict[addable[b]] & {addable[a]}):
                                    s.add(addable[a])
                                    s.add(addable[b])
                                    added = True
                                    break
                            if added:
                                break
                        if added:
                            improved_flag = True
                            break
                        else:
                            s.add(u1)
                            s.add(u2)
                    if improved_flag:
                        break

        return s

    best_solution = local_search_improve(best_solution, deadline - 10.0)

    # 4. Multiple random restarts of greedy + local search if time permits
    if not time_up(deadline - 15.0):
        for restart in range(80):
            if time_up(deadline - 5.0):
                break
            # Randomized greedy with some randomness in tie-breaking
            rem = set(candidates)
            greedy_set = set()
            while rem:
                # Pick random low-degree node
                low_deg_nodes = []
                min_d = float('inf')
                for node in rem:
                    d = len(conflict[node] & rem)
                    if d < min_d:
                        min_d = d
                        low_deg_nodes = [node]
                    elif d == min_d:
                        low_deg_nodes.append(node)
                chosen = random.choice(low_deg_nodes)
                greedy_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
            # Local search on this greedy result
            ls_result = local_search_improve(greedy_set, deadline - 5.0)
            if len(ls_result) > len(best_solution):
                best_solution = ls_result
>>>>>>> REPLACE

