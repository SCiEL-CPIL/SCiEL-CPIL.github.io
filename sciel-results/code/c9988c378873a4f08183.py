    # 3. Local Search / Improvement
    # Step A: Remove redundant bicliques
    changed = True
    while changed:
        changed = False
        for i in range(len(solution)):
            if len(solution) <= 1:
                break
            L_i, R_i = solution[i]
            edges_i = get_edges(L_i, R_i)
            other_edges = set()
            for j in range(len(solution)):
                if i == j:
                    continue
                L_j, R_j = solution[j]
                other_edges.update(get_edges(L_j, R_j))
            if edges_i.issubset(other_edges):
                solution.pop(i)
                changed = True
                break

    # Step B: Merge bicliques if possible
    merged = True
    while merged:
        merged = False
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                L_i, R_i = solution[i]
                L_j, R_j = solution[j]
                # Try merging L_i with L_j and R_i with R_j
                L_new = set(L_i) | set(L_j)
                R_new = set(R_i) | set(R_j)
                if is_biclique(L_new, R_new):
                    # Replace i and j with the merged biclique
                    solution.pop(j)
                    solution[i] = (list(L_new), list(R_new))
                    merged = True
                    break
            if merged:
                break
