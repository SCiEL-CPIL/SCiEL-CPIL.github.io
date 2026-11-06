import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adjacency matrix for fast lookup
    # adj[i][j] = 1 if edge exists
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1

    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges

    def find_all_k4s_and_i4s():
        """
        Finds all 4-cliques and 4-independent sets.
        Returns list of tuples of indices.
        """
        k4s = []
        i4s = []
        # O(n^4) is too slow for n=37? 37^4 ~ 1.8M. 
        # Checking each set of 4 takes O(1) or O(4^2) checks.
        # 1.8M * 16 checks ~ 30M ops. In Python, this might take a few seconds.
        # We can optimize by precomputing neighbors.
        
        # Let's use a simpler backtracking or just combinations for n<=42.
        # For n=37, combinations(37, 4) is 66045.
        # 66045 is very small. We can just iterate.
        
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            is_k4 = (adj[a][b] and adj[a][c] and adj[a][d] and 
                     adj[b][c] and adj[b][d] and adj[c][d])
            if is_k4:
                k4s.append(combo)
            
            is_i4 = (not adj[a][b] and not adj[a][c] and not adj[a][d] and 
                     not adj[b][c] and not adj[b][d] and not adj[c][d])
            if is_i4:
                i4s.append(combo)
                
        return k4s, i4s

    # Initial calculation
    k4s, i4s = find_all_k4s_and_i4s()
    
    # 2. Try to add vertices
    max_attempts_per_vertex = 20000
    max_new_vertices = 10 # Don't try to go too far if it's hard
    
    for new_v_idx in range(max_new_vertices):
        if time.time() - start_time > timeout:
            break
            
        # We want to find x in {0,1}^n such that:
        # For all k in k4s: sum(x[i] for i in k) < 4
        # For all i in i4s: sum(x[i] for i in i) > 0
        
        found = False
        # Randomized Local Search
        x = [random.randint(0, 1) for _ in range(n)]
        
        def count_violations(x):
            v_count = 0
            for k in k4s:
                s = sum(x[i] for i in k)
                if s == 4:
                    v_count += 1
            for i_set in i4s:
                s = sum(x[i] for i in i_set)
                if s == 0:
                    v_count += 1
            return v_count
            
        # Simulated Annealing / Hill Climbing
        current_violations = count_violations(x)
        best_x = x[:]
        best_violations = current_violations
        
        for iter in range(max_attempts_per_vertex):
            if time.time() - start_time > timeout:
                break
                
            if current_violations == 0:
                found = True
                best_x = x[:]
                break
                
            # Pick a random bit to flip
            idx = random.randint(0, n - 1)
            old_val = x[idx]
            x[idx] = 1 - old_val
            
            new_violations = count_violations(x)
            
            # Accept if better or equal (to escape plateaus, we could use SA, 
            # but simple hill climbing with random restarts might be enough 
            # given the low number of constraints relative to degrees of freedom)
            if new_violations <= current_violations:
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_x = x[:]
            else:
                # Revert
                x[idx] = old_val
                
            # Occasional random restart if stuck? 
            # For now, just keep going. If we hit a local minimum with >0 violations, 
            # we might fail. We can add a small probability to restart.
            if iter % 1000 == 0 and current_violations > 0:
                # Random restart
                x = [random.randint(0, 1) for _ in range(n)]
                current_violations = count_violations(x)
                
        if found:
            # Add the new vertex
            new_n = n + 1
            # Extend adj
            adj.append([0] * new_n)
            for i in range(n):
                adj[i].append(best_x[i])
                adj[new_n - 1][i] = best_x[i]
            adj[new_n - 1][new_n - 1] = 0 # No self loop
            
            n = new_n
            # Recalculate K4s and I4s
            k4s, i4s = find_all_k4s_and_i4s()
        else:
            # Failed to add a vertex
            break
            
    return n, get_edges()

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges.")
