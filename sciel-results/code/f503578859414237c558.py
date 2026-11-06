import random
import time

def construct():
    N = 40
    deadline = time.time() + 140.0
    
    best_edges = []
    best_count = 0
    
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Run multiple restarts
    # We can also try different random seeds implicitly by just shuffling
    
    while time.time() < deadline:
        A = [[0]*N for _ in range(N)]
        A2 = [[0]*N for _ in range(N)]
        edge_count = 0
        edges = []
        
        random.shuffle(pairs)
        
        for u, v in pairs:
            if A[u][v] == 1:
                continue
                
            # Check C6 condition
            # Path of length 4 exists if sum(A2[u][k] * A2[v][k]) > 0
            dot = 0
            Au = A2[u]
            Av = A2[v]
            for k in range(N):
                dot += Au[k] * Av[k]
                if dot > 0:
                    break
                    
            if dot == 0:
                # Add edge
                A[u][v] = 1
                A[v][u] = 1
                edges.append([u, v])
                edge_count += 1
                
                # Update A2
                Au_row = A[u]
                Av_row = A[v]
                A2u = A2[u]
                A2v = A2[v]
                for w in range(N):
                    A2u[w] += Av_row[w]
                    A2v[w] += Au_row[w]
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = edges
            
        # Early exit if we hit a very high number (e.g. > 110)
        if best_count > 110:
            break

    return best_edges
