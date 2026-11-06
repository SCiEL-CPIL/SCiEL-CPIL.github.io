    # Reset state if we detect a new instance (assuming 50 cities)
    if _tsp_state is not None and len(unvisited_nodes) == 50:
        _tsp_state = None
