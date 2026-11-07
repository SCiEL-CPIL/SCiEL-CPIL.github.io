def construct():
    n = 16
    half = n // 2
    left_partition = list(range(half))
    right_partition = list(range(half, n))
    
    result = []
    for a in left_partition:
        for b in right_partition:
            result.append([a, b])
    
    return result
