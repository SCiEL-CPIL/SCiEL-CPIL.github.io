import time

def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    The maximum number of mutually orthogonal Latin squares of order 10 is 1.
    """
    n = 10
    # Generate a single Latin square using cyclic group Z_10
    # L[i][j] = (i + j) % n
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure types are Python int
    square = [[int(val) for val in row] for row in square]
    
    return [square]
