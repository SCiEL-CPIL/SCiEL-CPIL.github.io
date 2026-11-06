    n = dag.get('n', None)
    parents = dag['parents']
    outputs = dag['outputs']
    inputs = dag.get('inputs', [])
    
    if n is None:
        # Infer n from parents and outputs
        # Assume parents is a list of lists of length n
        if isinstance(parents, list):
            n = len(parents)
        else:
            # Fallback
            max_idx = -1
            if isinstance(parents, dict):
                for p_list in parents.values():
                    for p in p_list:
                        if p > max_idx:
                            max_idx = p
            else:
                for p_list in parents:
                    for p in p_list:
                        if p > max_idx:
                            max_idx = p
            for o in outputs:
                if o > max_idx:
                    max_idx = o
            n = max_idx + 1
