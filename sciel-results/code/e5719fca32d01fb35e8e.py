    def count_4cliques_in_complement(verts, masks):
        """Count 4-cliques in the complement graph induced by verts.
        Two vertices a,b form a complement-edge iff NOT (masks[a] & (1<<b))."""
        nv = len(verts)
        if nv < 4:
            return 0
        count = 0
        for i in range(nv):
            a = verts[i]
            for j in range(i + 1, nv):
                b = verts[j]
                if masks[a] & (1 << b):
                    continue
                # Common complement-neighbors of a,b in verts
                # c must be non-adjacent to both a and b in G
                common = set()
                for k in range(j + 1, nv):
                    c = verts[k]
                    if not (masks[a] & (1 << c)) and not (masks[b] & (1 << c)):
                        common.add(c)
                clist = list(common)
                nc = len(clist)
                for k in range(nc):
                    c = clist[k]
                    for l in range(k + 1, nc):
                        d = clist[l]
                        if not (masks[c] & (1 << d)):
                            count += 1
        return count
