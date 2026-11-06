<<<<<<< SEARCH
    # Alternating degree-ordered search + general local search
    while time.time() < deadline:
        # Phase 1: Degree-ordered edge-replacement
        masks, deg, edges, current_score = degree_ordered_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        if time.time() > deadline:
            break
        
        # Phase 2: General local search for fine-tuning
        masks, deg, edges, current_score = local_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        # Random perturbation to escape local minima
        if time.time() > deadline:
            break
        
        masks_p = best_masks[:]
        deg_p = best_deg[:]
        edges_p = [e[:] for e in best_edges]
        for _ in range(4):
            ne = non_edges(masks_p)
            el = edge_list(masks_p)
            if not ne or not el:
                break
            add = random.choice(ne)
            rem = random.choice(el)
            if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
                continue
            masks_p[add[0]] |= (1 << add[1])
            masks_p[add[1]] |= (1 << add[0])
            masks_p[rem[0]] &= ~(1 << rem[1])
            masks_p[rem[1]] &= ~(1 << rem[0])
            deg_p[add[0]] += 1; deg_p[add[1]] += 1
            deg_p[rem[0]] -= 1; deg_p[rem[1]] -= 1
            for i, e in enumerate(edges_p):
                if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                    edges_p.pop(i)
                    break
            edges_p.append(list(add))
        
        masks_p, deg_p, edges_p, sc_p = degree_ordered_search(
            masks_p, deg_p, edges_p, score_graph(masks_p, deg_p), max_rounds=15000)
        masks_p, deg_p, edges_p, sc_p = local_search(
            masks_p, deg_p, edges_p, sc_p, max_rounds=15000)
        if sc_p < best_score:
            best_score = sc_p
            best_masks = masks_p[:]
            best_deg = deg_p[:]
            best_edges = [e[:] for e in edges_p]
=======
    # Alternating degree-ordered search + general local search
    while time.time() < deadline:
        # Phase 1: Degree-ordered edge-replacement
        masks, deg, edges, current_score = degree_ordered_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        if time.time() > deadline:
            break
        
        # Phase 2: General local search for fine-tuning
        masks, deg, edges, current_score = local_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        # Random perturbation to escape local minima
        if time.time() > deadline:
            break
        
        masks_p = best_masks[:]
        deg_p = best_deg[:]
        edges_p = [e[:] for e in best_edges]
        for _ in range(4):
            ne = non_edges(masks_p)
            el = edge_list(masks_p)
            if not ne or not el:
                break
            add = random.choice(ne)
            rem = random.choice(el)
            if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
                continue
            masks_p[add[0]] |= (1 << add[1])
            masks_p[add[1]] |= (1 << add[0])
            masks_p[rem[0]] &= ~(1 << rem[1])
            masks_p[rem[1]] &= ~(1 << rem[0])
            deg_p[add[0]] += 1; deg_p[add[1]] += 1
            deg_p[rem[0]] -= 1; deg_p[rem[1]] -= 1
            for i, e in enumerate(edges_p):
                if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                    edges_p.pop(i)
                    break
            edges_p.append(list(add))
        
        masks_p, deg_p, edges_p, sc_p = degree_ordered_search(
            masks_p, deg_p, edges_p, score_graph(masks_p, deg_p), max_rounds=15000)
        masks_p, deg_p, edges_p, sc_p = local_search(
            masks_p, deg_p, edges_p, sc_p, max_rounds=15000)
        if sc_p < best_score:
            best_score = sc_p
            best_masks = masks_p[:]
            best_deg = deg_p[:]
            best_edges = [e[:] for e in edges_p]
        
        # Final polish: wider neighborhood search on best solution
        if time.time() < deadline - 5:
            masks_f = best_masks[:]
            deg_f = best_deg[:]
            edges_f = [e[:] for e in best_edges]
            sc_f = score_graph(masks_f, deg_f)
            # Do a few rounds of wider local search
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                ne_f = non_edges(masks_f)
                el_f = edge_list(masks_f)
                ne_f_sorted = sorted(ne_f, key=lambda t: deg_f[t[0]] + deg_f[t[1]])
                el_f_sorted = sorted(el_f, key=lambda t: -(deg_f[t[0]] + deg_f[t[1]]))
                best_gain_f = 0
                best_add_f = None
                best_rem_f = None
                for add in ne_f_sorted[:35]:
                    u, v = add
                    for rem in el_f_sorted[:35]:
                        x, y = rem
                        if (u == x and v == y) or (u == y and v == x):
                            continue
                        masks_f[u] |= (1 << v)
                        masks_f[v] |= (1 << u)
                        masks_f[x] &= ~(1 << y)
                        masks_f[y] &= ~(1 << x)
                        deg_f[u] += 1; deg_f[v] += 1
                        deg_f[x] -= 1; deg_f[y] -= 1
                        ns_f = score_graph(masks_f, deg_f)
                        masks_f[u] &= ~(1 << v)
                        masks_f[v] &= ~(1 << u)
                        masks_f[x] |= (1 << y)
                        masks_f[y] |= (1 << x)
                        deg_f[u] -= 1; deg_f[v] -= 1
                        deg_f[x] += 1; deg_f[y] += 1
                        gain_f = sc_f - ns_f
                        if gain_f > best_gain_f:
                            best_gain_f = gain_f
                            best_add_f = add
                            best_rem_f = rem
                            if gain_f >= 2:
                                break
                    if best_gain_f >= 2:
                        break
                if best_add_f is None:
                    break
                u, v = best_add_f
                x, y = best_rem_f
                masks_f[u] |= (1 << v)
                masks_f[v] |= (1 << u)
                masks_f[x] &= ~(1 << y)
                masks_f[y] &= ~(1 << x)
                deg_f[u] += 1; deg_f[v] += 1
                deg_f[x] -= 1; deg_f[y] -= 1
                for i, e in enumerate(edges_f):
                    if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                        edges_f.pop(i)
                        break
                edges_f.append([u, v])
                sc_f -= best_gain_f
            if sc_f < best_score:
                best_score = sc_f
                best_masks = masks_f[:]
                best_deg = deg_f[:]
                best_edges = [e[:] for e in edges_f]
>>>>>>> REPLACE
