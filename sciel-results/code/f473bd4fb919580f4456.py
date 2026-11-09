    # Parse DAG structure
    if isinstance(dag, list):
        if len(dag) > 0 and isinstance(dag[0], dict):
            # List of dicts
            parents = {}
            for i, d in enumerate(dag):
                if 'parents' in d:
                    parents[i] = d['parents']
                else:
                    parents[i] = []
            
            # Try to find outputs
            # Check for 'is_output' or 'output' key
            outputs = []
            for i, d in enumerate(dag):
                if d.get('is_output', False) or d.get('output', False):
                    outputs.append(i)
            
            if not outputs:
                # Fallback to sinks
                all_parents_set = set()
                for ps in parents.values():
                    all_parents_set.update(ps)
                outputs = [i for i in parents.keys() if i not in all_parents_set]
                if not outputs:
                    outputs = [len(dag) - 1]
        else:
            # List of lists (or other sequences)
            parents = {i: list(dag[i]) for i in range(len(dag))}
            all_parents_set = set()
            for ps in parents.values():
                all_parents_set.update(ps)
            outputs = [i for i in parents.keys() if i not in all_parents_set]
            if not outputs:
                outputs = [len(dag) - 1]
    elif isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        if not outputs:
             all_parents_set = set()
             for ps in parents.values():
                 all_parents_set.update(ps)
             outputs = [i for i in parents.keys() if i not in all_parents_set]
    else:
        raise ValueError("Unsupported dag structure")
