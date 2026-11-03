    point_subs = [[] for _ in range(v)]
    for i, s in enumerate(all_subs):
        for p in s:
            point_subs[p].append(i)
