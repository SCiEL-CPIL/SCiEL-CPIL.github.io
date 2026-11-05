def construct():
    from math import isqrt
    from collections import defaultdict
    
    LIMIT = 10**9
    powerful = set()
    max_y = int(LIMIT ** (1/3)) + 1
    is_squarefree = [True] * (max_y + 1)
    for p in range(2, isqrt(max_y) + 1):
        p2 = p * p
        for j in range(p2, max_y + 1, p2):
            is_squarefree[j] = False
    squarefree_ys = [y for y in range(1, max_y + 1) if is_squarefree[y]]
    
    for y in squarefree_ys:
        y3 = y ** 3
        if y3 > LIMIT:
            break
        max_x = isqrt(LIMIT // y3)
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= LIMIT:
                powerful.add(n)
    
    powerful = sorted(powerful)
    
    sum_count = defaultdict(int)
    sum_pairs = defaultdict(list)
    
    for i, a in enumerate(powerful):
        for b in powerful[i:]:
            s = a + b
            sum_count[s] += 1
            sum_pairs[s].append([a, b])
    
    best_n = max(sum_count, key=sum_count.get)
    return {'N': best_n, 'pairs': sum_pairs[best_n]}
