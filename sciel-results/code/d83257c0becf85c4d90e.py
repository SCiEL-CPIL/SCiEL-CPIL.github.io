    def solve(available):
        # available is a set of ints
        if not available: return 0
        
        # Filter triples to those fully in available
        active_triples = [t for t in triples if t[0] in available and t[1] in available and t[2] in available]
        
        if not active_triples:
            return len(available)
            
        # Upper bound using greedy hitting set
        # We want to estimate min vertices to remove to kill all active_triples
        remaining_triples = set(active_triples)
        rem_count = 0
        avail_copy = set(available)
        # Simple greedy: pick vertex in most remaining triples
        while remaining_triples:
            # Find vertex with max frequency in remaining_triples
            freq = {}
            for t in remaining_triples:
                for v in t:
                    if v in avail_copy:
                        freq[v] = freq.get(v, 0) + 1
            if not freq: break
            v = max(freq, key=freq.get)
            rem_count += 1
            avail_copy.remove(v)
            remaining_triples = {t for t in remaining_triples if v not in t}
            
        ub = len(available) - rem_count
        
        if ub <= global_best:
            return ub # Prune
        
        # Branch on a triple
        # Heuristic: Pick triple with vertices having high degree in active_triples?
        # Or just the first one?
        # Let's pick the triple containing the vertex with highest degree in active_triples to force early resolution of high-impact vertices.
        deg = {}
        for t in active_triples:
            for v in t:
                deg[v] = deg.get(v, 0) + 1
        
        # Find vertex with max deg
        max_v = max(deg, key=deg.get)
        
        # Find a triple containing max_v
        branch_triple = None
        for t in active_triples:
            if max_v in t:
                branch_triple = t
                break
                
        if not branch_triple:
             branch_triple = active_triples[0]
             
        best = 0
        # Branch: remove a, b, or c
        for v in branch_triple:
            new_avail = available - {v}
            res = solve(new_avail)
            if res > best:
                best = res
                
        return best
    