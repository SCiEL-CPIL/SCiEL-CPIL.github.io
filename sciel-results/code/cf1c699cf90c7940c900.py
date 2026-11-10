"""
Constructs a single Latin square of order 10.
It is known that no pair of orthogonal Latin squares of order 10 exists.
Thus, the maximum number of mutually orthogonal Latin squares is 1.
"""

def construct():
    n = 10
    # Construct a standard cyclic Latin square
    # L[i][j] = (i + j) % n
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure all elements are Python ints (they already are from range arithmetic)
    # Ensure the structure is list of lists of ints
    result = []
    for row in square:
        result.append([int(x) for x in row])
        
    return [result]
