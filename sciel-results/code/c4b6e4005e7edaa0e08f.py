    # Parse the DAG structure - handle multiple possible formats
    if hasattr(dag, 'parents'):
        raw_parents = dag.parents
        if isinstance(raw_parents, dict):
            parents = {k: list(v) for k, v in raw_parents.items()}
        elif isinstance(raw_parents, (list, tuple)):
            # Assume list of (vertex, parents) pairs or similar
            parents = {}
            for item in raw_parents:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    v, ps = item[0], item[1]
                    parents[v] = list(ps)
                else:
                    # If it's just a list of vertices? Unlikely for 'parents' attr
                    # But let's be safe
                    pass
            # If the list was just vertices, we might need to infer parents? 
            # Usually 'parents' attr implies the mapping.
        else:
            parents = {}
        
        outputs = list(dag.outputs) if hasattr(dag, 'outputs') else []
        inputs = list(dag.inputs) if hasattr(dag, 'inputs') else []
        vertices = list(dag.vertices) if hasattr(dag, 'vertices') else list(parents.keys())
    elif isinstance(dag, dict):
        # ... existing code ...
