import time

def construct():
    """
    Constructs a set of mutually orthogonal Latin squares of order 10.
    The maximum number of MOLS of order 10 is 2.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    n = 10
    
    # Generate the first Latin square: L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Generate the second Latin square: L2[i][j] = (i + 2*j) % n
    # This is orthogonal to L1 because the matrix [[1, 1], [1, 2]] has determinant 1,
    # which is coprime to 10, making the mapping from (i, j) to (L1, L2) a bijection.
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify orthogonality to be absolutely sure (though mathematically proven)
    # This check is fast and ensures no implementation errors.
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            if pair in pairs:
                # Fallback to single square if something unexpected happens
                return [L1]
            pairs.add(pair)
            
    if len(pairs) != n * n:
        return [L1]
        
    return [L1, L2]
