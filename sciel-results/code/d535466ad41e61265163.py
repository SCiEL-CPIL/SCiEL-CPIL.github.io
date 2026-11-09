def get_strassen_2x2():
    tensors = []
    
    # 1
    u = [(0, 1), (3, 1)]
    v = [(0, 1), (3, 1)]
    w = [(0, 1), (3, 1)]
    tensors.append((u, v, w))
    
    # 2
    u = [(2, 1), (3, 1)]
    v = [(0, 1)]
    w = [(1, 1)]
    tensors.append((u, v, w))
    
    # 3
    u = [(0, 1)]
    v = [(1, 1), (3, -1)]
    w = [(2, 1)]
    tensors.append((u, v, w))
    
    # 4
    u = [(3, 1)]
    v = [(2, 1), (0, -1)]
    w = [(1, 1)]
    tensors.append((u, v, w))
    
    # 5
    u = [(0, 1), (1, 1)]
    v = [(3, 1)]
    w = [(0, -1), (2, 1)]
    tensors.append((u, v, w))
    
    # 6
    u = [(2, 1), (0, -1)]
    v = [(0, 1), (1, 1)]
    w = [(3, 1)]
    tensors.append((u, v, w))
    
    # 7
    u = [(1, 1), (3, -1)]
    v = [(2, 1), (3, 1)]
    w = [(0, 1)]
    tensors.append((u, v, w))
    
    return tensors
