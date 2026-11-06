<<<<<<< SEARCH
    # Normalize parents to dict v -> list
    if not isinstance(parents, dict):
        # Maybe it's a list of (v, parents) or list of lists
        parents = dict(parents)
=======
    # Normalize parents to dict v -> list
    if not isinstance(parents, dict):
        if isinstance(parents, (list, tuple)):
            # Assume list of lists, where index i corresponds to vertex i
            new_parents = {}
            for i, p_list in enumerate(parents):
                # Ensure p_list is a list of parents
                if isinstance(p_list, (list, tuple, set)):
                    new_parents[i] = list(p_list)
                else:
                    # If it's a single int, maybe it's a single parent?
                    new_parents[i] = [p_list]
            parents = new_parents
        else:
            # Fallback for other types, try dict conversion
            try:
                parents = dict(parents)
            except Exception:
                # If that fails, we can't parse the DAG structure
                return []
>>>>>>> REPLACE

Additionally, the code uses `from autoresearch_suite.science_tcs_expansion import pebble_baseline` which might not be available or might cause issues. It's better to return an empty list or a valid but suboptimal schedule if parsing fails, but since we are fixing the parse, let's ensure the parse is robust.

Also, `all_vertices` calculation:
