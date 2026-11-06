    def solve_sum_free(S):
        """
        Find a maximum sum-free subset of S by solving the Minimum Hitting Set
        problem on the hypergraph of sum conflicts.
        """
        n = len(S)
        if n <= 1:
            return S[:]
        
        s_set = set(S)
        
        # Generate conflicts
        # A conflict is a set of values that cannot all be present.
        # If a+b=c, then {a,b,c} is a conflict (if a!=b) or {a,c} (if a==b).
        conflicts = set()
        
        for c in S:
            for a in S:
                b = c - a
                if b in s_set:
                    if a == b:
                        conflict = frozenset([a, c])
                    else:
                        conflict = frozenset([a, b, c])
                    conflicts.add(conflict)
                    
        if not conflicts:
            return S[:]
            
        conflict_list = list(conflicts)
        
        # Greedy heuristic for initial bound
        greedy_removed = set()
        remaining = conflict_list[:]
        
        while remaining:
            counts = {}
            for conf in remaining:
                for v in conf:
                    counts[v] = counts.get(v, 0) + 1
            
            if not counts:
                break
                
            max_v = max(counts, key=counts.get)
            greedy_removed.add(max_v)
            
            remaining = [conf for conf in remaining if max_v not in conf]
            
        best_size = len(greedy_removed)
        best_removed = set(greedy_removed)
        
        # Branch and Bound
        def bb(conflicts_rem, removed):
            nonlocal best_removed, best_size
            
            if len(removed) >= best_size:
                return
                
            if not conflicts_rem:
                best_size = len(removed)
                best_removed = set(removed)
                return
                
            if len(removed) + 1 >= best_size:
                return
                
            target_conf = conflicts_rem[0]
            
            v_impact = {}
            for v in target_conf:
                impact = 0
                for other_conf in conflicts_rem:
                    if v in other_conf:
                        impact += 1
                v_impact[v] = impact
                
            candidates = sorted(target_conf, key=lambda x: v_impact[x], reverse=True)
            
            for v in candidates:
                new_removed = removed | {v}
                new_conflicts = [c for c in conflicts_rem if v not in c]
                bb(new_conflicts, new_removed)
                
        bb(conflict_list, set())
        
        result = [x for x in S if x not in best_removed]
        return result
