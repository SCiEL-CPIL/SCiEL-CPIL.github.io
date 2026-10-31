for idx in indices_list:
    conflict = False
    for nb in neighbors[idx]:
        if nb in selected_set:
            conflict = True
            break
    if not conflict:
        selected.append(idx)
        selected_set.add(idx)
