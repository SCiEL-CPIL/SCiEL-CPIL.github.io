    for v in vertices:
        if not parents.get(v, []):
            layer_of[v] = 0
        else:
            layer_of[v] = max(layer_of[p] for p in parents[v]) + 1
