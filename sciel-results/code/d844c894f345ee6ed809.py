def try_greedy(n):
    N = [ [set() for _ in range(5)] for _ in range(n) ]
    colors = [0] * (n * (n - 1) // 2)
    
    # Generate all edges (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)
    
    # Map edge to index for output
    # Index for (i, j) is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simply keep a dict or calculate on the fly.
    # For n=26, calculating index is fast.
    
    for i, j in edges:
        min_size = 2
        best_cs = []
        for c in range(5):
            size = len(N[i][c].intersection(N[j][c]))
            if size < min_size:
                min_size = size
                best_cs = [c]
            elif size == min_size:
                best_cs.append(c)
        
        if min_size > 1:
            return False, None
        
        c = random.choice(best_cs)
        
        # Calculate index
        # Number of edges before row i: (n-1) + (n-2) + ... + (n-i)
        # Sum = i * (n - 1 + (n - i)) / 2 = i * (2n - i - 1) / 2
        # Offset in row i: j - i - 1
        idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
        colors[idx] = c
        
        N[i][c].add(j)
        N[j][c].add(i)
        
    return True, colors
