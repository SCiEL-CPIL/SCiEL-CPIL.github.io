import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = i
            idx += 1
    
    # Safety: return fallback if somehow past deadline (shouldn't happen)
    if time.time() - start_time > 85:
        return [0] * NUM_EDGES
    
    return colors
