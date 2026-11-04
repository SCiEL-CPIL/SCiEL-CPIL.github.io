<<<<<<< SEARCH
    T0 = 1.0
    T_end = 0.01
    num_iterations = 20000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(50):
        if time.time() > deadline - 25:
            break

        x = greedy_init(restart)
        current_size = eval_state(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            add_candidates, remove_candidates = get_candidates(x)

            if not add_candidates and not remove_candidates:
                break

            move_type = py_rng.random()
            if move_type < 0.5 and add_candidates:
                v = py_rng.choice(add_candidates)
                if x[v]:
                    continue
                in_set = np.where(x)[0]
                if len(in_set) > 0 and np.any(conflict_D[v, in_set]):
                    continue
                x[v] = True
                delta = 1
                new_size = current_size + delta
                if new_size > best_current_size:
                    best_current_size = new_size
                    best_x = x.copy()
                current_size = new_size
            elif move_type < 0.8 and remove_candidates:
                v = py_rng.choice(remove_candidates)
                if not x[v]:
                    continue
                x[v] = False
                delta = -1
                new_size = current_size + delta
                if py_rng.random() < np.exp(delta / T):
                    current_size = new_size
                else:
                    x[v] = True
            else:
                if not add_candidates or not remove_candidates:
                    continue
                v_remove = py_rng.choice(remove_candidates)
                v_add = py_rng.choice(add_candidates)
                if v_remove == v_add:
                    continue
                if not x[v_remove] or x[v_add]:
                    continue
                in_set = np.where(x)[0]
                in_set_no_remove = [v for v in in_set if v != v_remove]
                if len(in_set_no_remove) > 0 and np.any(conflict_D[v_add, in_set_no_remove]):
                    continue
                x[v_remove] = False
                x[v_add] = True
                current_size = current_size

            T *= cooling

        if best_current_size > best_size_D:
            best_size_D = best_current_size
            best_support_set_D = set(np.where(best_x)[0].tolist())
=======
    T0 = 1.0
    T_end = 0.01
    num_iterations = 50000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(80):
        if time.time() > deadline - 25:
            break

        x = greedy_init(restart)
        current_size = eval_state(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            add_candidates, remove_candidates = get_candidates(x)

            if not add_candidates and not remove_candidates:
                break

            move_type = py_rng.random()
            if move_type < 0.5 and add_candidates:
                v = py_rng.choice(add_candidates)
                if x[v]:
                    continue
                in_set = np.where(x)[0]
                if len(in_set) > 0 and np.any(conflict_D[v, in_set]):
                    continue
                x[v] = True
                delta = 1
                new_size = current_size + delta
                if new_size > best_current_size:
                    best_current_size = new_size
                    best_x = x.copy()
                current_size = new_size
            elif move_type < 0.8 and remove_candidates:
                v = py_rng.choice(remove_candidates)
                if not x[v]:
                    continue
                x[v] = False
                delta = -1
                new_size = current_size + delta
                if py_rng.random() < np.exp(delta / T):
                    current_size = new_size
                else:
                    x[v] = True
            else:
                if not add_candidates or not remove_candidates:
                    continue
                v_remove = py_rng.choice(remove_candidates)
                v_add = py_rng.choice(add_candidates)
                if v_remove == v_add:
                    continue
                if not x[v_remove] or x[v_add]:
                    continue
                in_set = np.where(x)[0]
                in_set_no_remove = [v for v in in_set if v != v_remove]
                if len(in_set_no_remove) > 0 and np.any(conflict_D[v_add, in_set_no_remove]):
                    continue
                x[v_remove] = False
                x[v_add] = True
                current_size = current_size

            T *= cooling

        if best_current_size > best_size_D:
            best_size_D = best_current_size
            best_support_set_D = set(np.where(best_x)[0].tolist())
>>>>>>> REPLACE

