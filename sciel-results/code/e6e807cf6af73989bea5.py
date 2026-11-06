import numpy as np
from itertools import combinations
import time

def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses the algebraic construction based on F9 x Z3 (Lazebnik-Woldar style).
    """
    
    # --- Finite Field F9 Arithmetic ---
    # Representation: integer 0-8 mapping to (a, b) in F3 x F3
    # alpha^2 = 2
    def to_poly(x):
        return (x % 3, x // 3)
        
    def from_poly(a, b):
        return a % 3 + b * 3

    def f9_add(x, y):
        ax, bx = to_poly(x)
        ay, by = to_poly(y)
        return from_poly((ax + ay) % 3, (bx + by) % 3)

    def f9_sub(x, y):
        ax, bx = to_poly(x)
        ay, by = to_poly(y)
        return from_poly((ax - ay) % 3, (bx - by) % 3)

    def f9_mul(x, y):
        ax, bx = to_poly(x)
        ay, by = to_poly(y)
        # (a + b*alpha)(c + d*alpha) = ac + 2bd + (ad + bc)*alpha
        real = (ax * ay + 2 * bx * by) % 3
        imag = (ax * by + bx * ay) % 3
        return from_poly(real, imag)

    def f9_neg(x):
        return f9_sub(0, x)

    # --- Vertex Set ---
    # V = F9 x {0, 1, 2}
    # We will use 26 vertices. Let's use all 27 and remove index 0 (0,0).
    # Or just use the first 26.
    # Let's map vertex index i to (x, y)
    # i = x * 3 + y
    # We skip i=0.
    # So vertices are i in 1..26.
    n = 26
    vertices = []
    for i in range(1, 27):
        x = i // 3
        y = i % 3
        vertices.append((x, y))
        
    # --- Edge Types ---
    # An edge between (x1, y1) and (x2, y2) is characterized by (d, k) = (x1-x2, y1-y2)
    # Due to symmetry, (d, k) and (-d, -k) are the same undirected edge type.
    # We define a canonical representative for the pair { (d,k), (-d,-k) }
    
    def get_type(x1, y1, x2, y2):
        d = f9_sub(x1, x2)
        k = (y1 - y2) % 3
        
        d_neg = f9_neg(d)
        k_neg = (-k) % 3
        
        # Canonicalize: choose the lexicographically smaller of (d, k) and (d_neg, k_neg)
        # Represent (d, k) as d * 3 + k
        val1 = d * 3 + k
        val2 = d_neg * 3 + k_neg
        
        if val1 <= val2:
            return val1
        else:
            return val2

    # Precompute types for all pairs in the 26-vertex set
    # Map type_val (0..26) to type_id (0..12)
    # Note: (0,0) is not an edge type (self-loop), so val 0 is not used for edges.
    # However, d=0, k=1 -> val 1. d=0, k=2 -> val 2.
    # d=1, k=0 -> val 3. d_neg=8, k_neg=0 -> val 24.
    # We just collect all unique type values.
    
    type_map = {}
    type_ids = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            t_val = get_type(x1, y1, x2, y2)
            if t_val not in type_map:
                type_map[t_val] = len(type_ids)
                type_ids.append(t_val)
            
    num_types = len(type_ids)
    # Should be 13 types
    
    # --- Backtracking Search for Color Assignment ---
    # We need to assign colors 0..4 to the 13 types.
    # Constraint: No monochromatic C4.
    # A C4 in the vertex graph corresponds to a cycle of length 4 in the "type graph"?
    # Actually, we can just check the vertex graph directly for a candidate assignment.
    # But checking C4 for every assignment is expensive.
    # However, num_types is 13. 5^13 is too big.
    # We can use a greedy backtracking with pruning.
    # Pruning: If assigning color c to type t creates a C4 in the partial graph?
    # No, C4 involves 4 edges.
    
    # Alternative: Use the known property that the construction works.
    # I will use a simple local search / backtracking to find ONE valid assignment.
    # Since the space is small (13 vars), we can try random restarts or systematic search.
    
    import random
    random.seed(42)
    
    best_assignment = None
    
    def check_c4_free(color_of_type):
        # Build the full color array for the 26 vertices
        # color[i][j]
        col = [[-1]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = vertices[i]
                x2, y2 = vertices[j]
                t_val = get_type(x1, y1, x2, y2)
                t_id = type_map[t_val]
                c = color_of_type[t_id]
                col[i][j] = c
                col[j][i] = c
                
        # Check for monochromatic C4
        # A C4 is u-v-w-x-u.
        # We can check for any color c, if the subgraph of edges with color c has a C4.
        # Efficient check: For each color c, build adjacency matrix A_c.
        # C4 exists if there are u, v such that they have 2 common neighbors in A_c.
        # i.e., (A_c @ A_c)[u][v] >= 2 for u != v?
        # No, (A^2)[u][v] is the number of common neighbors.
        # If (A^2)[u][v] >= 2, then there is a C4 (u-a-v-b-u or similar).
        # Actually, if u and v have 2 common neighbors a, b, then u-a-v-b-u is a C4.
        
        for c in range(5):
            # Build adjacency matrix for color c
            # Use numpy for speed
            A = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(n):
                    if col[i][j] == c:
                        A[i][j] = 1
            # Count common neighbors
            # A @ A gives number of paths of length 2.
            # Diagonal is degree. Off-diagonal (i,j) is # common neighbors.
            # If any off-diagonal element >= 2, we have a C4.
            M = A @ A
            # Check off-diagonals
            # np.fill_diagonal(M, 0)
            # if np.max(M) >= 2: return False
            # But we must be careful: A[i][j]=1 implies edge.
            # Common neighbors of i and j.
            # If count >= 2, C4 exists.
            # We can just check max of M where i != j.
            # For n=26, this is fast.
            M_diag_zero = M.copy()
            np.fill_diagonal(M_diag_zero, 0)
            if np.max(M_diag_zero) >= 2:
                return False
        return True

    # Backtracking / Random Search
    # Try a few random assignments first
    for _ in range(1000):
        assignment = [random.randint(0, 4) for _ in range(num_types)]
        if check_c4_free(assignment):
            best_assignment = assignment
            break
            
    if best_assignment is None:
        # Try systematic backtracking
        # This might be slow if 1000 random fails, but 13 vars is small.
        # Let's do a simple DFS with pruning if possible, or just more randoms.
        # Given the structure, a solution likely exists.
        for _ in range(5000):
            assignment = [random.randint(0, 4) for _ in range(num_types)]
            if check_c4_free(assignment):
                best_assignment = assignment
                break

    if best_assignment is None:
        # Fallback: Return n=1 if we can't find it (should not happen)
        return {'n': 1, 'colors': []}
        
    # Generate the final color list
    colors_list = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            t_val = get_type(x1, y1, x2, y2)
            t_id = type_map[t_val]
            colors_list.append(best_assignment[t_id])
            
    return {'n': n, 'colors': colors_list}
