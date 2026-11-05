import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    # Precompute all K4s
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        k4s.append((edge_idx(a,b), edge_idx(a,c), edge_idx(a,d), edge_idx(b,c), edge_idx(b,d), edge_idx(c,d)))
        
    def count_violations(colors):
        # Returns number of K4s with < 5 colors
        violations = 0
        for e in k4s:
            c_set = set(colors[i] for i in e)
            if len(c_set) < 5:
                violations += 1
        return violations

    def count_colors(colors):
        return len(set(colors))

    # Try to find a 13-coloring
    best_score = float('inf')
    best_colors = None
    
    # Initialize with a random 13-coloring
    random.seed(42)
    max_colors = 13
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Local Search (Simulated Annealing)
        temp = 10.0
        current_score = count_violations(colors)
        
        if current_score == 0:
            return colors
            
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            new_color = random.randint(0, max_colors - 1)
            
            if old_color == new_color:
                continue
                
            colors[idx] = new_color
            new_score = count_violations(colors)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_score = new_score
                if current_score == 0:
                    return colors
            else:
                colors[idx] = old_color
                
            temp *= 0.99
            
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            
    if best_score == 0 and best_colors is not None:
        return best_colors
        
    # Fallback to 17-coloring if 13 not found
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
