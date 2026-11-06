def all_4cycles(adj):
    cycles = []
    seen = set()
    verts = list(range(n))
    for a in verts:
        for b in adj[a]:
            if b <= a: continue
            for c in adj[b]:
                if c <= a: continue
                if c in adj[a]: continue
                for d in adj[c]:
                    if d <= a: continue
                    if d in adj[b]: continue
                    if d not in adj[a]: continue
                    # a-b-c-d-a is a 4-cycle
                    # canonicalize
                    cyc = (a,b,c,d)
                    # generate all rotations and reversals, take min
                    rots = []
                    for r in range(4):
                        t = (cyc[r], cyc[(r+1)%4], cyc[(r+2)%4], cyc[(r+3)%4])
                        rots.append(t)
                        rots.append(t[::-1])
                    key = min(rots)
                    if key not in seen:
                        seen.add(key)
                        cycles.append((a,b,c,d))
    return cycles
