import time

def construct():
    # Try algebraic construction for n=26
    n = 26
    colors = try_algebraic_26()
    if colors is not None:
        return {'n': n, 'colors': colors}
    
    # Fallback to a known small valid coloring, e.g., n=11
    # For n=11, a simple cyclic coloring might work?
    # Or just use the previous fallback which was n=11.
    # Let's ensure the fallback is valid.
    # The previous fallback was: [min(j - i, n - j + i) - 1 ...]
    # This is a circulant coloring.
    # For n=11, max distance is 5. Colors 0..4.
    # Does this avoid C4?
    # In a circulant graph $C_{11}(1,2,3,4,5)$, does color 1 (edges diff 1 or 10) contain C4?
    # Edges of diff 1 form a cycle C11. No C4.
    # Edges of diff 2 form two cycles C11. No C4.
    # So this fallback is valid for n=11.
    
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_algebraic_26():
    # Vertices: (x, a) where x in 0..12, a in 0..1
    # Map vertex index v to (x, a): v = x * 2 + a? Or v = a * 13 + x?
    # Let's use v = x * 2 + a.
    # x = v // 2, a = v % 2.
    
    # Primitive root 2 mod 13
    # Powers of 2:
    # 1: 2
    # 2: 4
    # 3: 8
    # 4: 3
    # 5: 6
    # 6: 12
    # 7: 11
    # 8: 9
    # 9: 5
    # 10: 10
    # 11: 7
    # 12: 1
    
    # Sets S_i based on k % 4
    # k=1,5,9 -> d=2,6,5 -> S1
    # k=2,6,10 -> d=4,12,10 -> S2
    # k=3,7,11 -> d=8,11,7 -> S3
    # k=4,8,12 -> d=3,9,1 -> S4
    
    S1 = {2, 6, 5}
    S2 = {4, 12, 10}
    S3 = {8, 11, 7}
    S4 = {3, 9, 1}
    
    sets = [S1, S2, S3, S4]
    
    # We need to assign colors 0..4.
    # Color 0 is for d=0.
    # Colors 1..4 for d!=0.
    
    # Try different permutations of assigning sets to colors and e-values.
    # Structure: Color c gets edges from (Set_i, e=0) and (Set_j, e=1).
    # We need a bijection between Colors 1..4 and the pairs (i, j).
    # Actually, each color must cover exactly half the edges of d!=0?
    # Total edges d!=0 is 312. 312/4 = 78 edges per color.
    # Each Set has 3 values of d.
    # For a fixed d, there are 26 edges (13 with e=0, 13 with e=1).
    # So a Set of 3 d's has 3 * 26 = 78 edges.
    # If we take ALL edges from one Set (both e=0 and e=1), that's 78 edges.
    # This matches the count!
    # So, maybe Color 1 gets ALL edges where d in S1.
    # Color 2 gets ALL edges where d in S2.
    # Color 3 gets ALL edges where d in S3.
    # Color 4 gets ALL edges where d in S4.
    
    # Let's test this hypothesis.
    # If Color 1 contains all edges with d in S1, does it contain a C4?
    # The subgraph is the union of cycles for each d in S1.
    # For each d in S1, we have two C13s (e=0) and one C26 (e=1).
    # Union of these might contain C4.
    
    # Let's try this simple mapping first.
    def get_color_simple(d, e):
        if d == 0:
            return 0
        for idx, s in enumerate(sets):
            if d in s:
                return idx + 1
        return -1 # Should not happen

    colors = generate_colors(26, get_color_simple)
    if is_valid(colors, 26):
        return colors

    # If simple mapping fails, try mixing e.
    # Color 1: d in S1, e=0; d in S2, e=1
    # Color 2: d in S2, e=0; d in S3, e=1
    # Color 3: d in S3, e=0; d in S4, e=1
    # Color 4: d in S4, e=0; d in S1, e=1
    
    def get_color_mixed(d, e):
        if d == 0:
            return 0
        if e == 0:
            for idx, s in enumerate(sets):
                if d in s:
                    return idx + 1
        else: # e == 1
            # Shift sets
            # S1->1, S2->2, S3->3, S4->4 for e=0
            # For e=1, map S1 to Color 4? (from above: S1,e=1 -> Color 4)
            # S2,e=1 -> Color 1
            # S3,e=1 -> Color 2
            # S4,e=1 -> Color 3
            if d in sets[0]: return 4
            if d in sets[1]: return 1
            if d in sets[2]: return 2
            if d in sets[3]: return 3
        return -1

    colors = generate_colors(26, get_color_mixed)
    if is_valid(colors, 26):
        return colors
        
    return None

def generate_colors(n, color_func):
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            x_i, a_i = divmod(i, 2)
            x_j, a_j = divmod(j, 2)
            d = (x_i - x_j) % 13
            e = (a_i - a_j) % 2
            c = color_func(d, e)
            colors.append(c)
    return colors

def is_valid(colors, n):
    # Reconstruct adjacency
    neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            neighbors[c][i].add(j)
            neighbors[c][j].add(i)
            idx += 1
            
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                # Check if u and v have 2 common neighbors in color c
                # Common neighbors: N_c(u) intersect N_c(v)
                common = neighbors[c][u].intersection(neighbors[c][v])
                if len(common) > 1:
                    return False
    return True
