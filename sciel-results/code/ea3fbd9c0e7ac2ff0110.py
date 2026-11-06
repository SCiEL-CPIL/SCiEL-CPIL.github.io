import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # F4 arithmetic
    # Elements: 0, 1, 2 (w), 3 (w^2)
    # w^2 = w + 1 = 2 + 1 = 3
    # w^3 = 1
    # Add table (XOR-like but with specific mapping)
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1, 2+1=3
    # 3+3=0, 3+2=1, 3+1=2
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    mult_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]

    def add(a, b):
        return add_table[a][b]

    def mul(a, b):
        return mult_table[a][b]

    # Generate points of PG(2, 4)
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize to first non-zero being 1
                if x != 0:
                    inv = 1 if x == 1 else (3 if x == 2 else 2)
                    points.append((1, mul(y, inv), mul(z, inv)))
                elif y != 0:
                    inv = 1 if y == 1 else (3 if y == 2 else 2)
                    points.append((0, 1, mul(z, inv)))
                else:
                    inv = 1 if z == 1 else (3 if z == 2 else 2)
                    points.append((0, 0, 1))
    
    # Unique and sort
    points = sorted(list(set(points)))
    
    n = 20
    selected_points = points[:n]
    
    edges = []
    for i in range(n):
        p1 = selected_points[i]
        x1, y1, z1 = p1
        for j in range(i + 1, n):
            p2 = selected_points[j]
            x2, y2, z2 = p2
            
            # Bilinear form for polarity
            # B(P, Q) = x1x2 + y1y2 + z1z2 + x1y2 + y1z2 + z1x2
            val = 0
            val = add(val, mul(x1, x2))
            val = add(val, mul(y1, y2))
            val = add(val, mul(z1, z2))
            val = add(val, mul(x1, y2))
            val = add(val, mul(y1, z2))
            val = add(val, mul(z1, x2))
            
            if val == 0:
                edges.append([i, j, 0])
                
    return edges
