        assignments = {}
        assignments[root] = None  # to be assigned
        
        def backtrack(idx):
            ...
            if idx == k:
                return True
            
            v = order[idx]
            ...
            candidates = [h for h in range(20) if h not in used_hosts]
            rng.shuffle(candidates)
            
            for h in candidates:
                if p != -1:
                    ph = assignments[p]
                    ...
                assignments[v] = h
                if backtrack(idx + 1):
                    return True
                del assignments[v]
            ...
