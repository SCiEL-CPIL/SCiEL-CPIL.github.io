import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 10
    
    # Map index 0..9 to Z_5 x Z_2
    # idx = 2*a + b, where a in 0..4, b in 0..1
    def idx_to_group(idx):
        return (idx // 2, idx % 2)
    
    def group_to_idx(g):
        a, b = g
        return 2*a + b
        
    def group_add(g1, g2):
        return ((g1[0] + g2[0]) % 5, (g1[1] + g2[1]) % 2)
        
    def group_mul_scalar(g, k):
        # k is small integer
        res = (0, 0)
        for _ in range(k):
            res = group_add(res, g)
        return res

    squares = []
    
    # L_1: i + j
    sq1 = [[0]*n for _ in range(n)]
    for i in range(n):
        gi = idx_to_group(i)
        for j in range(n):
            gj = idx_to_group(j)
            val = group_add(gi, gj)
            sq1[i][j] = group_to_idx(val)
            
    # L_2: i + 2j
    sq2 = [[0]*n for _ in range(n)]
    for i in range(n):
        gi = idx_to_group(i)
        for j in range(n):
            gj = idx_to_group(j)
            val = group_add(gi, group_mul_scalar(gj, 2))
            sq2[i][j] = group_to_idx(val)
            
    squares.append(sq1)
    squares.append(sq2)
    
    # Verify
    def is_latin(square):
        for row in square:
            if set(row) != set(range(n)):
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if set(col) != set(range(n)):
                return False
        return True

    def are_orthogonal(a, b):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    if not is_latin(sq1) or not is_latin(sq2):
        return [[[(i+j)%10 for j in range(10)] for i in range(10)]]
        
    if not are_orthogonal(sq1, sq2):
        return [[[(i+j)%10 for j in range(10)] for i in range(10)]]
        
    return squares
