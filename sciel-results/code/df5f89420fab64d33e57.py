import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Try Paley tournaments for primes p ≡ 3 (mod 4)
    primes = [31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in primes:
        if time.time() > deadline - 10:
            break
        
        squares = set()
        for x in range(1, p):
            squares.add(x*x % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                if (j-i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        # Check validity
        adj = [[False]*p for _ in range(p)]
        idx = 0
        for i in range(p):
            for j in range(i+1, p):
                if orientation[idx] == 0:
                    adj[i][j] = True
                else:
                    adj[j][i] = True
                idx += 1
        
        valid = True
        for a in range(p):
            if not valid:
                break
            for b in range(a+1, p):
                if not valid:
                    break
                for c in range(b+1, p):
                    if not valid:
                        break
                    for d in range(c+1, p):
                        dominated = False
                        for v in range(p):
                            if v in (a,b,c,d):
                                continue
                            if adj[v][a] and adj[v][b] and adj[v][c] and adj[v][d]:
                                dominated = True
                                break
                        if not dominated:
                            valid = False
                            break
        
        if valid:
            return {'n': p, 'orientation': orientation}
    
    # Fallback to Paley 83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add(x*x % p)
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j-i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
