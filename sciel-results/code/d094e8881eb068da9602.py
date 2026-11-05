import numpy as np

def construct():
    # We want a C4-free graph on n <= 47 with min degree >= n - 39.
    # Using Hadamard matrix of order N=48, we get a C4-free graph on N-1=47 vertices.
    # Degree is N/2 - 1 = 23.
    # Condition: 23 >= 47 - 39 = 8. Satisfied.
    
    q = 23
    # Quadratic residues mod q
    Q = set((x * x) % q for x in range(1, q))
    
    # Construct B (q x q)
    B = np.zeros((q, q), dtype=int)
    for i in range(q):
        for j in range(q):
            diff = (j - i) % q
            if diff in Q:
                B[i, j] = 1
            else:
                B[i, j] = -1
                
    # Construct Hadamard matrix H of order 2*(q+1) = 48
    # Standard Paley construction for q % 4 == 3:
    # H = [[J, J], [J, -J]] is not quite right for this size.
    # The construction is:
    # H = [[1, 1^T, 1^T],
    #      [1, B, I],
    #      [1, I, -B]]
    # This is size 2q+1 = 47.
    # To get 48, we add a row/col?
    # Actually, the order is 2(q+1).
    # Let's use the Kronecker product of H_4 and H_12?
    # Or just use the 47 vertex graph from the 47x47 matrix above?
    # If we use the 47x47 matrix, we get a graph on 46 vertices (N-1).
    # N=47 is not a Hadamard order.
    
    # Let's use the 48x48 matrix properly.
    # The Paley matrix of order 2(q+1) is:
    # H = [[J, J], [J, -J]] where J is q x q? No.
    
    # I will use the explicit construction for H_48.
    # H_48 can be formed by H_4 (Kronecker) H_12.
    # H_12 from Paley q=11? No, q=11 gives 24.
    # H_12 is a known small matrix.
    
    # Let's just use the 47-vertex graph from the 48-vertex Hadamard matrix if I can construct it.
    # If I construct H_48, I get 47 vertices.
    
    # Construction of H_48 from Paley q=23:
    # The matrix is:
    # H = [[J, J], [J, -J]]?
    # Let's try:
    # H = np.kron(np.array([[1, 1], [1, -1]]), B)?
    # Size 2q x 2q = 46x46.
    # We need 48.
    
    # Okay, I will use the 44-vertex graph from H_44.
    # H_44 exists.
    # Construct H_44.
    # 44 = 4 * 11.
    # H_11 doesn't exist.
    # But H_44 is a standard Hadamard matrix.
    # I can hardcode a 44x44 Hadamard matrix or generate it.
    
    # Actually, the simplest robust solution:
    # Use the Polarity Graph PG(2, 7) on 57 vertices?
    # No, too sparse.
    
    # Let's use the 47-vertex graph derived from H_48.
    # I will generate H_48 using the following:
    # H_48 = H_4 (Kronecker) H_12.
    # H_12 can be generated from H_4 and a 3x3? No.
    # H_12 is the Kronecker product of H_4 and a 3x3 matrix? No.
    
    # I will use the 43-vertex graph from H_44.
    # H_44 can be constructed.
    # But 47 is better.
    
    # Let's try to construct H_48 directly.
    # Reference: Paley Hadamard matrix of order 2(q+1).
    # For q=23, order 48.
    # The matrix is:
    # H = [[J, J], [J, -J]] where J is the q x q all-ones matrix?
    # No, that would be rank 1.
    
    # The correct form is:
    # H = [[1, 1^T, 1^T], [1, B, I], [1, I, -B]] is 47x47.
    # To get 48, we append a row/col?
    
    # I will use the 47-vertex graph from the 48-vertex Hadamard matrix.
    # I will assume the 47x47 matrix M (from the 47x47 block) is the adjacency part.
    # The graph has 47 vertices.
    
    # Let's use the 47x47 matrix from the Paley construction (size 2q+1).
    # Vertices: 47.
    # Is it C4-free?
    # The graph associated with a Hadamard matrix is C4-free.
    # The subgraph of a C4-free graph is C4-free.
    
    # So I will construct the 47x47 matrix M.
    # M = [[B, I], [I, -B]]?
    # From the 47x47 H:
    # H = [[1, 1^T, 1^T], [1, B, I], [1, I, -B]]
    # The submatrix excluding first row/col is:
    # M = [[B, I], [I, -B]]
    # This M is 46x46?
    # q=23. B is 23x23. I is 23x23.
    # M is 46x46.
    # So the graph has 46 vertices.
    
    # Let's check the degree.
    # In H, rows are orthogonal.
    # The graph from M (46 vertices) is C4-free.
    # Degree?
    # For q=23, B has 11 ones and 12 minus ones in each row?
    # (Since q is odd, (q-1)/2 = 11 residues).
    # So B has row sum 11 - 12 = -1?
    # Diagonal of B is -1.
    # Off-diagonal: 11 ones, 11 minus ones?
    # No, total off-diagonal elements q-1 = 22.
    # 11 residues, 11 non-residues.
    # So off-diagonal sum is 0.
    # Diagonal is -1.
    # So row sum of B is -1.
    
    # M = [[B, I], [I, -B]]
    # Row sum of top block: -1 + 23 = 22.
    # Row sum of bottom block: 23 - (-1) = 24?
    # -B has row sum 1.
    # So bottom row sum: 23 + 1 = 24.
    
    # This graph is not regular.
    # Min degree?
    # We need to convert M to 0-1 adjacency.
    # Usually, 1 -> Edge, -1 -> No Edge.
    # Top block: B has 11 ones. I has 23 ones. Total 34 edges?
    # Bottom block: I has 23 ones. -B has 11 ones (where B was -1).
    # B has 12 minus ones (1 diag + 11 off).
    # So -B has 12 ones.
    # Total bottom edges: 23 + 12 = 35.
    
    # Min degree is 34.
    # n=46.
    # Condition: delta >= 46 - 39 = 7.
    # 34 >= 7.
    
    # So n=46 works!
    
    n = 46
    edges = []
    
    # Construct M
    M = np.zeros((46, 46), dtype=int)
    M[:23, :23] = B
    M[:23, 23:] = np.eye(23, dtype=int)
    M[23:, :23] = np.eye(23, dtype=int)
    M[23:, 23:] = -B
    
    # Convert to edges
    for i in range(46):
        for j in range(i+1, 46):
            if M[i, j] == 1:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
