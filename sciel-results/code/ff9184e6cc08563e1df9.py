import random
import time

def count_colors(colors, quad):
    # quad is a tuple of 4 indices
    i, j, k, l = quad
    c = {colors[i][j], colors[i][k], colors[i][l], colors[j][k], colors[j][l], colors[k][l]}
    return len(c)

def is_valid(colors):
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if count_colors(colors, (i, j, k, l)) < 5:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n = 16
    quads = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    quads.append((i, j, k, l))
    
    # Try k from 13 to 15
    for k in range(13, 16):
        # Random initialization
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, k-1)
                colors[j][i] = colors[i][j]
        
        # Check initial
        if is_valid(colors):
            # Flatten and return
            res = []
            for i in range(n):
                for j in range(i+1, n):
                    res.append(colors[i][j])
            return res

        # Local Search
        # We want to minimize the number of "bad" quads (those with < 5 colors)
        # Or simply maximize the minimum color count?
        # Let's define score as sum over all quads of (1 if count < 5 else 0). Minimize this.
        
        def get_violations(colors):
            v = 0
            for q in quads:
                if count_colors(colors, q) < 5:
                    v += 1
            return v

        current_violations = get_violations(colors)
        
        max_iters = 50000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_c = colors[i][j]
            
            # Try changing to a random new color
            new_c = random.randint(0, k-1)
            if new_c == old_c:
                continue
                
            colors[i][j] = new_c
            colors[j][i] = new_c
            
            new_violations = get_violations(colors)
            
            if new_violations <= current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    res = []
                    for a in range(n):
                        for b in range(a+1, n):
                            res.append(colors[a][b])
                    return res
            else:
                # Revert
                colors[i][j] = old_c
                colors[j][i] = old_c
                
        # If we didn't find a solution for this k, try next k
        # To improve search, we could use simulated annealing, but simple hill climbing might work if k is close to optimal.
        
    # Fallback: Return the seed solution (17 colors) if no better found
    # Or return the best found so far? The task says "Minimize". 
    # If we fail to find a valid one with k<17, we should return a valid one.
    # The seed is valid.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
