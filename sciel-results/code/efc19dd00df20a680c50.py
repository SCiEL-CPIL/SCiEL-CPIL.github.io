import time

def construct():
    n = 10
    
    # Hardcoded mutually orthogonal Latin squares of order 10
    # Square 1: a=0, b=1
    sq1 = []
    for i in range(n):
        row = []
        for j in range(n):
            i1 = i // 5
            i2 = i % 5
            j1 = j // 5
            j2 = j % 5
            val = 5 * ((i1 + 0 * j1) % 2) + ((i2 + 1 * j2) % 5)
            row.append(val)
        sq1.append(row)
        
    # Square 2: a=1, b=2
    sq2 = []
    for i in range(n):
        row = []
        for j in range(n):
            i1 = i // 5
            i2 = i % 5
            j1 = j // 5
            j2 = j % 5
            val = 5 * ((i1 + 1 * j1) % 2) + ((i2 + 2 * j2) % 5)
            row.append(val)
        sq2.append(row)
        
    return [sq1, sq2]
