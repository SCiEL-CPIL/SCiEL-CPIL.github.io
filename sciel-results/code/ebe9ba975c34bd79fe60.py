import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # Gives (q+1)-regular graph of girth >= d+5 with (q+1)*q^d vertices
    # For k=3: q+1=3, so q=2
    # For g=16: need d+5>=16, so d>=11, d must be odd
    q = k - 1  # q=2 for k=3
    # d must be odd and >= g-5
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    if d % 2 == 0:
        d += 1  # ensure odd
    
    # Precompute rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute powers
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
    
    total_ids = q ** d  # 2^11 = 2048
    
    # Precompute digit tables
    digits_flat = [0] * (total_ids * d)
    for vid in range(total_ids):
        tmp = vid
        for pos in range(1, d + 1):
            digits_flat[vid * d + (pos - 1)] = tmp % q
            tmp //= q
    
    def get_digit(val, pos):
        return digits_flat[val * d + (pos - 1)]
    
    # Precompute neighbors
    point_neighbors = [[] for _ in range(total_ids)]
    for p_id in range(total_ids):
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = (p_c + l_a * p_b) % q
                current_l_id += l_c * powers[c-1]
            point_neighbors[p_id].append(current_l_id)
    
    line_neighbors = [[] for _ in range(total_ids)]
    for l_id in range(total_ids):
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = (l_c - l_a * p_b) % q
                current_p_id += p_c * powers[c-1]
            line_neighbors[l_id].append(current_p_id)
    
    # BFS
    n_vertices = 2 * total_ids
    index = [-1] * n_vertices
    order = []
    edges = []
    
    start_vertex = 0
    index[start_vertex] = 0
    order.append(start_vertex)
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side = vertex // total_ids
        vid = vertex % total_ids
        
        if side == 0:
            for l_id in point_neighbors[vid]:
                n_vertex = total_ids + l_id
                if index[n_vertex] == -1:
                    index[n_vertex] = len(order)
                    order.append(n_vertex)
                edges.append((index[vertex], index[n_vertex]))
        else:
            for p_id in line_neighbors[vid]:
                n_vertex = p_id
                if index[n_vertex] == -1:
                    index[n_vertex] = len(order)
                    order.append(n_vertex)
                edges.append((index[vertex], index[n_vertex]))
    
    n = len(order)
    return n, edges
