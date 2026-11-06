    # Parse the DAG
    try:
        if isinstance(dag, dict):
            parents_raw = dag.get('parents', {})
            outputs_raw = dag.get('outputs', [])
            vertices_raw = dag.get('vertices', None)
            inputs_raw = dag.get('inputs', None)
        else:
            parents_raw = getattr(dag, 'parents', {})
            outputs_raw = getattr(dag, 'outputs', [])
            vertices_raw = getattr(dag, 'vertices', None)
            inputs_raw = getattr(dag, 'inputs', None)
            
        # Robustly convert parents to dict
        if isinstance(parents_raw, dict):
            parents = {k: (v if isinstance(v, list) else [v]) for k, v in parents_raw.items()}
        elif isinstance(parents_raw, (list, tuple)):
            parents = {}
            for item in parents_raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    k, v = item
                    parents[k] = v if isinstance(v, list) else [v]
                # Handle case where it's a flat list? Unlikely for DAGs.
        else:
            parents = {}

        outputs = list(outputs_raw) if outputs_raw else []
        
        if vertices_raw is not None:
            all_vertices = set(vertices_raw)
        else:
            all_vertices = set(parents.keys())
            for v, ps in parents.items():
                for p in ps:
                    all_vertices.add(p)
            for o in outputs:
                all_vertices.add(o)

        if inputs_raw is not None:
            input_vertices = set(inputs_raw)
        else:
            input_vertices = set(v for v in all_vertices if not parents.get(v, []))
            
    except Exception as e:
        # If parsing fails completely, we can't do much.
        # But the task says "Fix it". The error was a ValueError.
        # We'll let it raise if it's a real data error, but the previous code raised.
        # The goal is to prevent the ValueError from dict() on bad input.
        raise
