<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = 16
    half = n // 2
    edges = []
    
    for u in range(half):
        if time.time() > deadline:
            break
        for v in range(half, n):
            if time.time() > deadline:
                break
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
