import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Finite field F4 arithmetic
    # F4 = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # We represent elements as 0, 1, 2 (alpha), 3 (alpha+1)
    # Addition table (XOR): 0^0=0, 0^1=1, 0^2=2, 0^3=3, 1^1=0, 1^2=3, 1^3=2, 2^2=0, 2^3=1, 3^3=0
    # Multiplication table:
    # 0*x=0
    # 1*x=x
    # 2*2=3, 2*3=2, 3*3=1
    # So: mul[2][2]=3, mul[2][3]=2, mul[3][2]=2, mul[3][3]=1
    
    # Vertices are points in F4^2, represented as (a, b) where a, b in {0,1,2,3}
    # Vertex index = a * 4 + b
    
    # For each pair of vertices, compute the "slope" (direction) of the line
    # Direction is the normalized difference vector
    # Two directions are equivalent if they are scalar multiples
    
    # First, let's identify all distinct directions in F4^2
    # A direction is an equivalence class of nonzero vectors under scalar multiplication
    # Nonzero vectors: 15 total (16 - 1 for zero)
    # Each line through origin has 3 nonzero points, so 15/3 = 5 distinct directions
    
    # Let's enumerate the 5 directions:
    # Direction 0: vectors where x=0 (vertical): (0,1), (0,2), (0,3)
    # Direction 1: vectors where y=0 (horizontal): (1,0), (2,0), (3,0)
    # Direction 2: vectors where y=x: (1,1), (2,3), (3,2)  [since 2*1=2, 2*2=3, 3*1=3, 3*3=1]
    #   Check: (1,1)*2 = (2, 2*1) = (2,2)? No, 2*1=2 in F4? Wait, let me recheck.
    #   Actually in F4: 0,1,2,3 with 2^2=3, 2*3=2, 3^2=1
    #   (1,1)*2 = (2, 2) but 2 is not a valid "slope" representation. Let me just enumerate all 15 nonzero vectors and group them.
    
    # Better approach: for each edge, compute the direction and assign a color based on it
    # Then check if we need more colors or can merge some directions
    
    # Let's compute the direction for each pair of points
    # Direction of (p1, p2) is the equivalence class of (p2-p1) under scalar multiplication
    
    # First, build a mapping from vector to direction class
    # We'll pick a canonical representative for each direction
    
    # All 15 nonzero vectors in F4^2:
    # (1,0), (2,0), (3,0) -> direction 0 (horizontal)
    # (0,1), (0,2), (0,3) -> direction 1 (vertical)
    # (1,1), (1,2), (1,3)
    # (2,1), (2,2), (2,3)
    # (3,1), (3,2), (3,3)
    
    # Let's group them by checking scalar multiples
    def get_direction(dx, dy):
        """Return the direction class of vector (dx, dy) in F4^2"""
        if dx == 0:
            return 1  # vertical
        if dy == 0:
            return 0  # horizontal
        # For nonzero dx and dy, compute dy/dx
        # In F4, division by x is multiplication by x^3 = x^{-1}
        # Inverses: 1^{-1}=1, 2^{-1}=3 (since 2*3=2? No, 2*3=2 in my table? Let me fix)
        # Actually: 1*1=1, 1*2=2, 1*3=3
        # 2*1=2, 2*2=3, 2*3=2  -- this is wrong. Let me redo F4 multiplication.
        # F4 = GF(4) with polynomial x^2 + x + 1
        # Elements: 0, 1, alpha, alpha+1
        # alpha^2 = alpha + 1, alpha^3 = 1
        # So: 0, 1, 2(alpha), 3(alpha+1)
        # Multiplication:
        # 0*x = 0
        # 1*x = x
        # 2*1 = 2
        # 2*2 = 2*alpha = alpha^2 = alpha+1 = 3
        # 2*3 = 2*(alpha+1) = 2*alpha + 2 = 3 + 2 = 1 (since 2+3 = alpha + alpha+1 = 1)
        # 3*1 = 3
        # 3*2 = 1
        # 3*3 = (alpha+1)^2 = alpha^2 + 2*alpha + 1 = (alpha+1) + 0 + 1 = alpha = 2
        # So the table is:
        #   0 1 2 3
        # 0: 0 0 0 0
        # 1: 0 1 2 3
        # 2: 0 2 3 1
        # 3: 0 3 1 2
        
        # So 2^{-1} = 3 (since 2*3=1), 3^{-1} = 2 (since 3*2=1)
        
        # dy/dx = dy * dx^{-1}
        if dx == 1:
            ratio = dy
        elif dx == 2:
            ratio = dy * 3  # 2^{-1} = 3
        elif dx == 3:
            ratio = dy * 2  # 3^{-1} = 2
        else:
            ratio = 0  # shouldn't happen since dx != 0
        
        if ratio == 0:
            return 0  # horizontal (dy=0)
        elif ratio == 1:
            return 2  # slope 1
        elif ratio == 2:
            return 3  # slope 2
        elif ratio == 3:
            return 4  # slope 3
    
    # Now assign colors based on direction
    colors = [0] * num_edges
    for i in range(n):
        for j in range(i + 1, n):
            # Convert vertex indices to F4^2 coordinates
            ax, ay = i // 4, i % 4
            bx, by = j // 4, j % 4
            
            # Compute difference
            dx = bx ^ ax  # XOR is addition in F4
            dy = by ^ ay
            
            # Get direction
            dir_idx = get_direction(dx, dy)
            colors[edge_index[(i, j)]] = dir_idx
    
    # Check validity
    seed_valid = True
    min_colors_in_k4 = 16
    for k4_idx in range(num_k4s):
        e_list = k4_edges[k4_idx]
        distinct = len({colors[e] for e in e_list})
        if distinct < 5:
            seed_valid = False
            break
        min_colors_in_k4 = min(min_colors_in_k4, distinct)
    
    if not seed_valid:
        # Fallback to 1-factorization
        for k in range(15):
            e_idx = edge_index[(min(k, 15), max(k, 15))]
            colors[e_idx] = k
            for i in range(1, 8):
                u = (k + i) % 15
                v = (k - i) % 15
                if u > v:
                    u, v = v, u
                e_idx = edge_index[(u, v)]
                colors[e_idx] = k
    
    best_colors = colors[:]
    best_num_colors = max(colors) + 1
    
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            other_colors = set()
            for e in e_list:
                if e != e_idx:
                    other_colors.add(colors[e])
            
            if new_color in other_colors:
                if len(other_colors) < 5:
                    return False
            else:
                if len(other_colors) < 4:
                    return False
        return True

    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    # Local search to reduce colors if needed
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        for _ in range(1000):
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c, colors):
                    min_valid = c
                    break
            
            if min_valid != -1:
                colors[e_idx] = min_valid
                improved = True
        
        if improved:
            no_improve_count = 0
            if max(colors) < best_num_colors:
                compacted = compact_colors(colors)
                num_unique = max(compacted) + 1
                if num_unique < best_num_colors:
                    best_num_colors = num_unique
                    best_colors = compacted
                    colors = compacted
        else:
            no_improve_count += 1
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            found_kick = False
            max_kick_color = best_num_colors + 5
            for _ in range(20):
                new_c = random.randint(0, max_kick_color)
                if new_c == current_c:
                    continue
                if can_change_edge(e_idx, new_c, colors):
                    colors[e_idx] = new_c
                    found_kick = True
                    break
            
        iter_count += 1
        if iter_count % 5000 == 0:
            compacted = compact_colors(colors)
            num_unique = max(compacted) + 1
            if num_unique < best_num_colors:
                best_num_colors = num_unique
                best_colors = compacted
            colors = compacted

    final_compact = compact_colors(best_colors)
    return final_compact
