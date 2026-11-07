import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 10
    
    # Represent elements as (a, b) in Z_5 x Z_2
    # idx = 5*a + b where a in 0..4, b in 0..1
    def idx_to_group(idx):
        return (idx // 2, idx % 2)
    
    def group_to_idx(g):
        a, b = g
        return 2*a + b
        
    def group_add(g1, g2):
        return ((g1[0] + g2[0]) % 5, (g1[1] + g2[1]) % 2)
        
    def group_sub(g1, g2):
        return ((g1[0] - g2[0]) % 5, (g1[1] - g2[1]) % 2)
        
    def group_mul_scalar(g, k):
        k_mod5 = k % 5
        k_mod2 = k % 2
        return ((g[0] * k_mod5) % 5, (g[1] * k_mod2) % 2)

    def make_square(multiplier):
        """Create Latin square L[i][j] = i + multiplier * j"""
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            gi = idx_to_group(i)
            for j in range(n):
                gj = idx_to_group(j)
                val = group_add(gi, group_mul_scalar(gj, multiplier))
                sq[i][j] = group_to_idx(val)
        return sq

    def is_latin(square):
        for row in square:
            if len(set(row)) != n:
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if len(set(col)) != n:
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

    def pairwise_orthogonal(squares_list):
        for i in range(len(squares_list)):
            for j in range(i+1, len(squares_list)):
                if not are_orthogonal(squares_list[i], squares_list[j]):
                    return False
        return True

    # Try multipliers that give orthogonal squares
    # L_k[i][j] = i + k*j
    # Two such squares with multipliers k1, k2 are orthogonal iff k1 != k2 (mod appropriate structure)
    
    squares = []
    used_multipliers = set()
    
    # Try all multipliers from 0 to 9
    for mult in range(n):
        sq = make_square(mult)
        if not is_latin(sq):
            continue
        # Check orthogonality with all existing squares
        orthogonal = True
        for existing in squares:
            if not are_orthogonal(sq, existing):
                orthogonal = False
                break
        if orthogonal:
            squares.append(sq)
            used_multipliers.add(mult)
            if len(squares) >= 9:
                break
    
    # If we couldn't find even 2, fall back to basic construction
    if len(squares) < 2:
        sq1 = make_square(1)
        sq2 = make_square(2)
        if is_latin(sq1) and is_latin(sq2) and are_orthogonal(sq1, sq2):
            return [sq1, sq2]
        else:
            return [[[(i+j)%10 for j in range(10)] for i in range(10)]]
    
    return squares
