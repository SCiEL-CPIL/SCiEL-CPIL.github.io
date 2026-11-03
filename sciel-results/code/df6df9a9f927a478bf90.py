    def greedy_min_degree():
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            # Pick random from min-degree candidates
            min_node = candidates[np.random.randint(len(candidates))]
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s
    