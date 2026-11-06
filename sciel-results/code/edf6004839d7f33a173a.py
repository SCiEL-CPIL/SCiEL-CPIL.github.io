<<<<<<< SEARCH
    # --- 1. Parse DAG Structure ---
    # Assume dag is a list where dag[i] is a list of parent indices for vertex i.
    # Or a list of tuples (vertex_id, parents, is_output).
    # We will try to detect the format.
    
    n_vertices = len(dag)
    parents = [set() for _ in range(n_vertices)]
    children = [set() for _ in range(n_vertices)]
    is_output = [False] * n_vertices
    
    # Detect format
    if n_vertices > 0:
        first_item = dag[0]
        if isinstance(first_item, (list, tuple, set)):
            # Format: dag[i] is list of parents
            for i in range(n_vertices):
                p_list = dag[i]
                parents[i] = set(p_list)
                for p in parents[i]:
                    children[p].add(i)
            # Outputs are typically the last vertices or specified separately.
            # In many DAG pebbling problems, outputs are vertices with no children.
            # Let's assume outputs are vertices with no children.
            for i in range(n_vertices):
                if not children[i]:
                    is_output[i] = True
        elif isinstance(first_item, tuple) and len(first_item) >= 3:
            # Format: dag[i] = (id, parents, is_output)
            for i in range(n_vertices):
                item = dag[i]
                # item[0] is id, item[1] is parents, item[2] is output flag
                # But indices in dag list correspond to vertex indices usually.
                # Let's assume item[1] is the list of parents.
                p_list = item[1]
                parents[i] = set(p_list)
                is_output[i] = bool(item[2])
                for p in parents[i]:
                    children[p].add(i)
        elif isinstance(first_item, dict):
            # Format: dag[i] = {'parents': [...], 'output': bool}
            for i in range(n_vertices):
                item = dag[i]
                parents[i] = set(item.get('parents', []))
                is_output[i] = bool(item.get('output', False))
                for p in parents[i]:
                    children[p].add(i)
        else:
            # Fallback: Assume dag[i] is a list of parents
            for i in range(n_vertices):
                if isinstance(dag[i], (list, tuple, set)):
                    parents[i] = set(dag[i])
                    for p in parents[i]:
                        children[p].add(i)
            for i in range(n_vertices):
                if not children[i]:
                    is_output[i] = True
=======
    # --- 1. Parse DAG Structure ---
    # Handle both List and Dict inputs.
    # List: dag[i] is parents of i.
    # Dict: dag[v] is parents of v.
    
    if isinstance(dag, dict):
        # Map vertex IDs to 0..N-1 for internal processing
        vertex_ids = list(dag.keys())
        id_to_index = {vid: i for i, vid in enumerate(vertex_ids)}
        n_vertices = len(vertex_ids)
        
        parents = [set() for _ in range(n_vertices)]
        children = [set() for _ in range(n_vertices)]
        is_output = [False] * n_vertices
        
        for vid in vertex_ids:
            idx = id_to_index[vid]
            p_list = dag[vid]
            # p_list contains vertex IDs
            parent_indices = set(id_to_index[p] for p in p_list)
            parents[idx] = parent_indices
            for p_idx in parent_indices:
                children[p_idx].add(idx)
        
        # Determine outputs: vertices with no children
        for i in range(n_vertices):
            if not children[i]:
                is_output[i] = True
    else:
        # Assume list
        n_vertices = len(dag)
        parents = [set() for _ in range(n_vertices)]
        children = [set() for _ in range(n_vertices)]
        is_output = [False] * n_vertices
        
        for i in range(n_vertices):
            item = dag[i]
            if isinstance(item, (list, tuple, set)):
                parents[i] = set(item)
                for p in parents[i]:
                    children[p].add(i)
            elif isinstance(item, dict):
                parents[i] = set(item.get('parents', []))
                is_output[i] = bool(item.get('output', False))
                for p in parents[i]:
                    children[p].add(i)
            elif isinstance(item, tuple) and len(item) >= 3:
                parents[i] = set(item[1])
                is_output[i] = bool(item[2])
                for p in parents[i]:
                    children[p].add(i)
        
        # If no explicit output flags were set, assume sinks are outputs
        if not any(is_output):
            for i in range(n_vertices):
                if not children[i]:
                    is_output[i] = True
>>>>>>> REPLACE

The `KeyError: 0` was likely due to `dag[0]` when `dag` was a dict not containing key `0`. The fix handles dicts by remapping keys to indices.

Additionally, the `topo_order.index(x)` is inefficient and can be replaced with a precomputed `topo_index` dictionary to avoid O(N) lookups and potential issues if the list is large.

