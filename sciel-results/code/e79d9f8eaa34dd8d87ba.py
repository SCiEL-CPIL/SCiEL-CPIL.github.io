import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin

    def is_valid(n, adj):
        if n < 5:
            return True
        # Check K5
        for combo in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_indep = False
                    else:
                        is_clique = False
            if is_clique or is_indep:
                return False
        return True

    def get_edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges

    # 1. Try Paley(41)
    q = 41
    if q % 4 == 1:
        residues = {(x * x) % q for x in range(1, q)}
        adj_41 = [[False] * q for _ in range(q)]
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj_41[i][j] = True
                    adj_41[j][i] = True
        
        if is_valid(q, adj_41):
            return q, get_edges_from_adj(q, adj_41)

    # 2. Fall back to Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    
    if not is_valid(q, adj):
        # Should not happen, but just in case
        return 0, []

    best_n = q
    best_adj = [row[:] for row in adj]

    # 3. Try to extend using Simulated Annealing on the new vertex's neighborhood
    # We try to add vertices one by one up to 42
    
    current_n = q
    current_adj = [row[:] for row in adj]
    
    def get_k4s_and_indep_k4s(n, adj):
        k4s = []
        indep_k4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    if adj[combo[i]][combo[j]]:
                        is_indep = False
                    else:
                        is_clique = False
            if is_clique:
                k4s.append(combo)
            if is_indep:
                indep_k4s.append(combo)
        return k4s, indep_k4s

    for target_n in range(current_n + 1, 43):
        if time.time() > deadline:
            break
            
        # Get constraints from current graph
        k4s, indep_k4s = get_k4s_and_indep_k4s(current_n, current_adj)
        
        # We want to find N (neighborhood of new vertex) such that:
        # No K4 is subset of N
        # No I4 is subset of ~N
        
        found = False
        max_attempts = 50
        attempt = 0
        
        while not found and attempt < max_attempts and time.time() < deadline:
            attempt += 1
            
            # Initialize N
            # Heuristic: Random, or based on degrees?
            N = [random.choice([True, False]) for _ in range(current_n)]
            
            # Simulated Annealing
            T = 1.0
            cool_rate = 0.95
            min_T = 0.01
            steps = 2000
            
            for s in range(steps):
                if time.time() > deadline:
                    break
                
                # Find a random violated constraint
                # To speed up, we can sample a few K4s and I4s
                # But for n=37, total K4s is small. Let's just scan.
                
                viol_k4 = None
                viol_ik4 = None
                
                # Check K4s in N
                for k4 in k4s:
                    if all(N[v] for v in k4):
                        viol_k4 = k4
                        break
                        
                if not viol_k4:
                    # Check I4s in ~N
                    for ik4 in indep_k4s:
                        if not any(N[v] for v in ik4):
                            viol_ik4 = ik4
                            break
                
                if viol_k4 is None and viol_ik4 is None:
                    found = True
                    break
                
                # Choose a move
                # Flip a bit involved in the violation
                if viol_k4:
                    v = random.choice(viol_k4)
                    N[v] = not N[v]
                    # This move reduces cost by at least 1 (for this K4)
                    # But might create others.
                elif viol_ik4:
                    v = random.choice(viol_ik4)
                    N[v] = not N[v]
                else:
                    v = random.randint(0, current_n - 1)
                    N[v] = not N[v]
                
                # Simple SA: Always accept if we fixed a violation?
                # Actually, we don't calculate full cost.
                # Let's just do random flips with restarts if stuck.
                # This is basically a random walk with bias towards fixing violations.
                
                T *= cool_rate
                if T < min_T:
                    break
            
            if found:
                # Add vertex
                new_adj = [row[:] for row in current_adj]
                # Extend adj matrix
                for i in range(current_n):
                    new_adj[i].append(N[i])
                    new_adj[i][i] = False # Self loop false
                new_row = [False] * (current_n + 1)
                for i in range(current_n):
                    new_row[i] = N[i]
                new_row[current_n] = False
                new_adj.append(new_row)
                
                current_adj = new_adj
                current_n += 1
                best_n = current_n
                best_adj = [row[:] for row in current_adj]
                break # Move to next target_n

    return best_n, get_edges_from_adj(best_n, best_adj)
