def decompose_walk_to_simple_paths(walk):
    if len(walk) < 2:
        return [walk] if walk else []
        
    if len(walk) == len(set(walk)):
        return [walk]
        
    # Check if it's a simple cycle
    if walk[0] == walk[-1]:
        inner = walk[1:-1]
        if len(inner) == len(set(inner)):
            # Simple cycle
            return [walk[:-1], [walk[0], walk[-1]]]
            
    # Find repeated vertex
    seen = {}
    for i, v in enumerate(walk):
        if v in seen:
            start = seen[v]
            end = i
            cycle = walk[start:end+1]
            prefix = walk[:start]
            suffix = walk[end:]
            
            cycle_paths = decompose_walk_to_simple_paths(cycle)
            
            prefix_path = prefix + [v]
            suffix_path = [v] + suffix
            
            result = cycle_paths
            if len(prefix_path) > 1:
                result.append(prefix_path)
            if len(suffix_path) > 1:
                result.append(suffix_path)
                
            return result
        seen[v] = i
        
    return [walk]
