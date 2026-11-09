    for i, (u, v) in enumerate(edges):
        mask = 0
        near = {u, v}
        for x in (u, v):
            near |= adj.get(x, set())
        for y in near:
            for z in adj.get(y, ()):
                if z > y:
                    continue
                e2 = (min(y, z), max(y, z))
                if e2 in eidx and e2 != (u, v):
                    j = eidx[e2]
                    mask |= (1 << j)
        conflict[i] = mask
