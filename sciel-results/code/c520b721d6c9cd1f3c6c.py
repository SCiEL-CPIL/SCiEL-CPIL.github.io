import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n = 29
    # Calculate quadratic residues modulo 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        if time.time() > deadline:
             # Fallback to n=17 if somehow this takes too long (it won't)
             n = 17
             squares = set()
             for x in range(1, n):
                 squares.add((x * x) % n)
             colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
             return {'n': n, 'colors': colors}
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
