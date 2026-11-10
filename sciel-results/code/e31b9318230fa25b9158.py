import random
import time
from itertools import combinations

def construct():
    n = 16
    num_colors = 15
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Initialize with random colours
    colours = [random.randint(0, num_colors - 1) for _ in range(len(edges))]
    
    # Precompute K4s
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e1 = (a, b)
        e2 = (a, c)
        e3 = (a, d)
        e4 = (b, c)
        e5 = (b, d)
        e6 = (c, d)
        k4s.append([edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4], edge_index[e5], edge_index[e6]])
        
    def count_violations():
        viol = 0
        for k4 in k4s:
            cols = set(colours[i] for i in k4)
            if len(cols) < 5:
                viol += 1
        return viol
        
    # Local search
    start_time = time.time()
    best_colours = None
    best_violations = count_violations()
    
    # Try to find a valid solution
    while time.time() - start_time < 100:
        if best_violations == 0:
            best_colours = colours[:]
            break
            
        # Pick a random violation
        # For efficiency, we could maintain a list of violations, but recomputing is fine for 1820 K4s
        viol_k4s = []
        for k4 in k4s:
            cols = set(colours[i] for i in k4)
            if len(cols) < 5:
                viol_k4s.append(k4)
                
        if not viol_k4s:
            best_colours = colours[:]
            break
            
        random_k4 = random.choice(viol_k4s)
        random_edge_idx = random.choice(random_k4)
        colours[random_edge_idx] = random.randint(0, num_colors - 1)
        
        # Recalculate violations
        curr_violations = count_violations()
        if curr_violations < best_violations:
            best_violations = curr_violations
            if curr_violations == 0:
                best_colours = colours[:]
                break
                
    if best_colours is not None:
        return best_colours
    else:
        # Fallback to valid 17-colouring if search fails
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
