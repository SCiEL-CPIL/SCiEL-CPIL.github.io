        shifted = set()
        conflict = False
        for s in sums:
            ns = s + w
            if ns in sums:
                conflict = True
                break
            shifted.add(ns)
        
        if not conflict:
            new_sums = sums.union(shifted)
            included.append(idx)
            dfs(pos + 1, included, new_sums, count + 1, order)
            included.pop()
