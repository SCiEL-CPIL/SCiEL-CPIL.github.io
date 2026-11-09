<<<<<<< SEARCH
    # Calculate conflicts
    # Conflict is a pair of trees that share an edge in K20.
    # We want to minimize the number of conflicts.
    # Total edges in K20 is 190.
    # Total edges in trees is sum(k-1) for k=1..20 = 190.
    # So a perfect packing has 0 conflicts.
    
    def calc_conflicts(state):
        used_edges = set()
        conflicts = 0
        for i in range(20):
            k = i + 1
            if state[i] is None:
                continue
            mapping = state[i]
            if k == 1:
                continue
            tree_edges = trees[i]
            for u, v in tree_edges:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used_edges:
                    conflicts += 1
                else:
                    used_edges.add((e1, e2))
        return conflicts
    
    current_conflicts = calc_conflicts(state)
=======
    # Efficient conflict tracking
    # edge_count[e] = number of trees using edge e
    # conflicts = sum over all edges of max(0, count-1)
    
    def build_edge_index(state):
        """Build edge usage counts and return conflicts."""
        edge_count = {}
        conflicts = 0
        for i in range(20):
            k = i + 1
            if state[i] is None or k == 1:
                continue
            mapping = state[i]
            tree_edges = trees[i]
            for u, v in tree_edges:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                e = (e1, e2)
                edge_count[e] = edge_count.get(e, 0) + 1
                if edge_count[e] > 1:
                    conflicts += 1
        return edge_count, conflicts
    
    def remove_tree_edges(edge_count, i):
        """Remove all edges of tree i from edge_count, return delta_conflicts."""
        k = i + 1
        if state[i] is None or k == 1:
            return 0
        mapping = state[i]
        tree_edges = trees[i]
        delta = 0
        for u, v in tree_edges:
            e1 = min(mapping[u], mapping[v])
            e2 = max(mapping[u], mapping[v])
            e = (e1, e2)
            if e in edge_count:
                old_count = edge_count[e]
                if old_count > 1:
                    delta -= 1
                edge_count[e] -= 1
                if edge_count[e] == 0:
                    del edge_count[e]
        return delta
    
    def add_tree_edges(edge_count, i):
        """Add all edges of tree i to edge_count, return delta_conflicts."""
        k = i + 1
        if state[i] is None or k == 1:
            return 0
        mapping = state[i]
        tree_edges = trees[i]
        delta = 0
        for u, v in tree_edges:
            e1 = min(mapping[u], mapping[v])
            e2 = max(mapping[u], mapping[v])
            e = (e1, e2)
            old_count = edge_count.get(e, 0)
            edge_count[e] = old_count + 1
            if old_count >= 1:
                delta += 1
        return delta
    
    # Initialize edge tracking
    edge_count, current_conflicts = build_edge_index(state)
    best_state = [s[:] if s else None for s in state]
    best_conflicts = current_conflicts
    best_edge_count = dict(edge_count)
>>>>>>> REPLACE

