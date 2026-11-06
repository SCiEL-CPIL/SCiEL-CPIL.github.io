import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    half = 8
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k44(rows):
        for i in range(half):
            for j in range(i+1, half):
                for k in range(j+1, half):
                    for l in range(k+1, half):
                        mask = rows[i] & rows[j] & rows[k] & rows[l]
                        if popcount(mask) >= 4:
                            return True
        return False

    def count_edges(rows):
        return sum(popcount(r) for r in rows)

    best_rows = None
    best_score = 0

    # Fallback seed
    seed_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    for r in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization with p=0.5
        rows = [random.getrandbits(half) for _ in range(half)]
        
        # Make valid
        while has_k44(rows):
            i = random.randint(0, half-1)
            j = random.randint(0, half-1)
            if rows[i] & (1 << j):
                rows[i] &= ~(1 << j)
            else:
                # If no edge to remove, this shouldn't happen if we started with edges
                # But if rows are all 0, has_k44 is False.
                break
                
        # Local Search with Simulated Annealing-like moves
        current_rows = rows[:]
        current_score = count_edges(current_rows)
        
        # Simple Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = [(i, j) for i in range(half) for j in range(half) if not (current_rows[i] & (1 << j))]
            random.shuffle(missing)
            
            # Try a limited number of edges to avoid timeout
            for u, v in missing:
                if time.time() > deadline:
                    break
                old_val = current_rows[u]
                current_rows[u] |= (1 << v)
                
                if not has_k44(current_rows):
                    improved = True
                    current_score += 1
                    break
                else:
                    current_rows[u] = old_val
                    
        if current_score > best_score:
            best_score = current_score
            best_rows = current_rows[:]

    if best_rows is None or best_score <= len(seed_edges):
        return seed_edges

    edges = []
    for i in range(half):
        for j in range(half):
            if best_rows[i] & (1 << j):
                edges.append([i, j + half])
                
    return edges
