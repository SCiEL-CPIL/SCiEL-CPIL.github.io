import itertools
import time

def _has_mono_k222(n, colors):
    """
    Checks if the coloring of K_n contains a monochromatic K_{2,2,2}.
    colors: list of 0/1 for edges (i,j) with i<j in lexicographic order.
    """
    if n < 6:
        return False
        
    # Build adjacency matrix
    adj = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
            
    # Precompute partitions of 6 elements into 3 pairs
    # Indices 0..5
    partitions = [
        ((0, 1), (2, 3), (4, 5)), ((0, 1), (2, 4), (3, 5)), ((0, 1), (2, 5), (3, 4)),
        ((0, 2), (1, 3), (4, 5)), ((0, 2), (1, 4), (3, 5)), ((0, 2), (1, 5), (3, 4)),
        ((0, 3), (1, 2), (4, 5)), ((0, 3), (1, 4), (2, 5)), ((0, 3), (1, 5), (2, 4)),
        ((0, 4), (1, 2), (3, 5)), ((0, 4), (1, 3), (2, 5)), ((0, 4), (1, 5), (2, 3)),
        ((0, 5), (1, 2), (3, 4)), ((0, 5), (1, 3), (2, 4)), ((0, 5), (1, 4), (2, 3))
    ]
    
    for c in [0, 1]:
        for combo in itertools.combinations(range(n), 6):
            for part in partitions:
                p1, p2, p3 = part
                v1, v2 = combo[p1[0]], combo[p1[1]]
                w1, w2 = combo[p2[0]], combo[p2[1]]
                x1, x2 = combo[p3[0]], combo[p3[1]]
                
                # Check edges between P1 and P2
                if (adj[v1][w1] != c or adj[v1][w2] != c or 
                    adj[v2][w1] != c or adj[v2][w2] != c):
                    continue
                # Check edges between P2 and P3
                if (adj[w1][x1] != c or adj[w1][x2] != c or 
                    adj[w2][x1] != c or adj[w2][x2] != c):
                    continue
                # Check edges between P3 and P1
                if (adj[x1][v1] != c or adj[x1][v2] != c or 
                    adj[x2][v1] != c or adj[x2][v2] != c):
                    continue
                
                return True
    return False

def _get_paley_colors(n):
    """
    Generates Paley coloring for K_n where n is a prime (or prime power) 
    such that the quadratic residue graph is defined.
    Color 0 if (j-i) is a quadratic residue mod n, else 1.
    """
    if n < 2:
        return []
        
    # Find nonzero quadratic residues
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # Note: For n % 4 == 1, -1 is a residue, so graph is symmetric.
            # For n % 4 == 3, -1 is not a residue, but we define color based on (j-i).
            # The task requires a coloring of K_n, which is undirected.
            # The Paley graph for n%4==3 is directed if defined as i->j if j-i in QR.
            # However, the standard Paley graph is defined for n%4==3 as undirected?
            # Actually, for n%4==3, the Paley graph is usually defined on the field, 
            # and the adjacency is symmetric because -1 is a non-residue?
            # Wait. If x is a residue, is -x a residue?
            # If n%4==3, -1 is a non-residue. So if x is residue, -x is non-residue.
            # So (j-i) in QR does NOT imply (i-j) in QR.
            # Thus, the graph defined by "color 0 if diff in QR" is directed.
            # But we need an undirected coloring.
            # For n=17 (17%4=1), -1 is a residue. So it's symmetric.
            # For n=29 (29%4=1), -1 is a residue. So it's symmetric.
            # So Paley graphs for n%4==1 are valid undirected colorings.
            
            colors.append(0 if diff in squares else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=29
    n = 29
    colors = _get_paley_colors(n)
    if time.time() < deadline:
        if not _has_mono_k222(n, colors):
            return {'n': n, 'colors': colors}
            
    # Try n=17
    n = 17
    colors = _get_paley_colors(n)
    if time.time() < deadline:
        if not _has_mono_k222(n, colors):
            return {'n': n, 'colors': colors}
            
    # Fallback to n=1 if all else fails (should not happen)
    return {'n': 1, 'colors': []}
