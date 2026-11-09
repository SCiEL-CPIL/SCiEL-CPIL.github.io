import random
import time
import math

def construct():
    n = 16
    # Initialize K_8,8 + (0,1)
    A = [[0]*n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            A[u][v] = 1
            A[v][u] = 1
    A[0][1] = 1
    A[1][0] = 1
    
    # Use a set for fast edge lookup and a list for removal selection
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            edges_set.add((u, v))
    edges_set.add((0, 1))
    
    # List of edges for removal selection (indices correspond to edges_set contents)
    edge_list = list(edges_set)

    def get_score():
        degs = [0]*n
        for u, v in edges_set:
            degs[u] += 1
            degs[v] += 1
        
        max_sum = 0
        # Iterate over all triangles
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    for k in range(j+1, n):
                        if A[i][k] and A[j][k]:
                            s = degs[i] + degs[j] + degs[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = get_score()
    best_score = current_score
    best_A = [row[:] for row in A]
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    T = 10.0
    cooling = 0.9995
    
    # Pre-compute all possible pairs for faster random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit:
        # 2-Swap Move
        # Pick 2 distinct edges to remove
        idx1 = random.randint(0, len(edge_list) - 1)
        idx2 = random.randint(0, len(edge_list) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(edge_list) - 1)
        
        e1 = edge_list[idx1]
        e2 = edge_list[idx2]
        
        # Pick 2 distinct non-edges to add
        new_e1 = None
        new_e2 = None
        
        while True:
            p1 = all_pairs[random.randint(0, len(all_pairs)-1)]
            if p1 not in edges_set:
                new_e1 = p1
                break
            
        while True:
            p2 = all_pairs[random.randint(0, len(all_pairs)-1)]
            if p2 not in edges_set and p2 != new_e1:
                new_e2 = p2
                break
        
        # Perform move
        # Remove e1, e2
        edges_set.remove(e1)
        edges_set.remove(e2)
        A[e1[0]][e1[1]] = 0
        A[e1[1]][e1[0]] = 0
        A[e2[0]][e2[1]] = 0
        A[e2[1]][e2[0]] = 0
        
        # Add new_e1, new_e2
        edges_set.add(new_e1)
        edges_set.add(new_e2)
        A[new_e1[0]][new_e1[1]] = 1
        A[new_e1[1]][new_e1[0]] = 1
        A[new_e2[0]][new_e2[1]] = 1
        A[new_e2[1]][new_e2[0]] = 1
        
        # Update edge_list to reflect changes
        edge_list[idx1] = new_e1
        edge_list[idx2] = new_e2
        
        new_score = get_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = [row[:] for row in A]
        else:
            delta = new_score - current_score
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    edges_set.discard(new_e1)
                    edges_set.discard(new_e2)
                    edges_set.add(e1)
                    edges_set.add(e2)
                    
                    A[new_e1[0]][new_e1[1]] = 0
                    A[new_e1[1]][new_e1[0]] = 0
                    A[new_e2[0]][new_e2[1]] = 0
                    A[new_e2[1]][new_e2[0]] = 0
                    
                    A[e1[0]][e1[1]] = 1
                    A[e1[1]][e1[0]] = 1
                    A[e2[0]][e2[1]] = 1
                    A[e2[1]][e2[0]] = 1
                    
                    edge_list[idx1] = e1
                    edge_list[idx2] = e2
            else:
                # Revert
                edges_set.discard(new_e1)
                edges_set.discard(new_e2)
                edges_set.add(e1)
                edges_set.add(e2)
                
                A[new_e1[0]][new_e1[1]] = 0
                A[new_e1[1]][new_e1[0]] = 0
                A[new_e2[0]][new_e2[1]] = 0
                A[new_e2[1]][new_e2[0]] = 0
                
                A[e1[0]][e1[1]] = 1
                A[e1[1]][e1[0]] = 1
                A[e2[0]][e2[1]] = 1
                A[e2[1]][e2[0]] = 1
                
                edge_list[idx1] = e1
                edge_list[idx2] = e2
        
        T *= cooling

    # Reconstruct edges from best_A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i][j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
