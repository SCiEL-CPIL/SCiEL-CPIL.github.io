import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    pos = {}
    for i, (u, v) in enumerate(edges):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
        
    def edge_cost(u, v):
        return pos[(u, v)]
        
    def max_cost(cyc):
        m = 0
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            c = edge_cost(u, v)
            if c > m:
                m = c
        return m
        
    def apply_2opt(cycle, i, j):
        """Reverse segment [i+1..j] in cycle"""
        cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
        
    def compute_2opt_max(cycle, i, j):
        """Compute max edge cost after 2-opt move at positions i, j"""
        u1 = cycle[i]
        v1 = cycle[(i + 1) % n]
        u2 = cycle[j]
        v2 = cycle[(j + 1) % n]
        
        nc1 = edge_cost(u1, u2)
        nc2 = edge_cost(v1, v2)
        
        max_remaining = 0
        for k in range(n):
            if k == i or k == j:
                continue
            u = cycle[k]
            v = cycle[(k + 1) % n]
            c = edge_cost(u, v)
            if c > max_remaining:
                max_remaining = c
                
        return max(max_remaining, nc1, nc2)
        
    def oropt_move(cycle, i, length):
        """Move segment of given length starting at position i to position j"""
        seg = cycle[i:i+length]
        remaining = cycle[:i] + cycle[i+length:]
        return seg, remaining
        
    def compute_oropt_max(cycle, i, j, length):
        """Compute max edge cost after Or-opt move: move segment [i..i+length-1] before position j"""
        if length == 1:
            # Move single vertex i before position j
            if i == j or abs(i - j) == 1 or (i == 0 and j == n-1) or (j == 0 and i == n-1):
                return float('inf')
            
            u_prev = cycle[(i-1) % n]
            u = cycle[i]
            u_next = cycle[(i+1) % n]
            
            v_before = cycle[(j-1) % n]
            v_after = cycle[j]
            
            # Remove u from its position, insert before j
            # Edges affected: (u_prev, u_next) replaces (u_prev, u) and (u, u_next)
            # (v_before, u) and (u, v_after) replace (v_before, v_after)
            e1 = edge_cost(u_prev, u_next)
            e2 = edge_cost(v_before, u)
            e3 = edge_cost(u, v_after)
            
            max_remaining = 0
            for k in range(n):
                if k == i or k == j or k == (j-1) % n or k == (i-1) % n or k == (i+1) % n:
                    continue
                u_k = cycle[k]
                v_k = cycle[(k + 1) % n]
                # Skip if this edge involves the moved vertex
                if u_k == u or v_k == u:
                    continue
                c = edge_cost(u_k, v_k)
                if c > max_remaining:
                    max_remaining = c
                    
            return max(max_remaining, e1, e2, e3)
        else:
            return float('inf')
            
    best_cycle = list(range(n))
    best_score = max_cost(best_cycle)
    
    # Try multiple restarts
    rng_search = random.Random(123)
    
    for restart in range(150):  # More restarts with faster convergence
        if time.time() > deadline:
            break
            
        if restart == 0:
            cycle = list(range(n))
        else:
            cycle = list(range(n))
            rng_search.shuffle(cycle)
            
        current_max = max_cost(cycle)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_move = None
            best_move_type = None
            best_new_max = current_max
            
            # 2-opt moves
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                        
                    new_max = compute_2opt_max(cycle, i, j)
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (i, j)
                        best_move_type = '2opt'
            
            # Or-opt moves (length 1)
            for i in range(n):
                for j in range(n):
                    if i == j or abs(i - j) == 1 or (i == 0 and j == n-1) or (j == 0 and i == n-1):
                        continue
                        
                    new_max = compute_oropt_max(cycle, i, j, 1)
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (i, j)
                        best_move_type = 'oropt'
            
            if best_move is not None:
                if best_move_type == '2opt':
                    i, j = best_move
                    apply_2opt(cycle, i, j)
                elif best_move_type == 'oropt':
                    i, j = best_move
                    # Move vertex i before position j
                    u = cycle[i]
                    cycle.pop(i)
                    # Adjust j if needed
                    if i < j:
                        j -= 1
                    if j >= n:
                        j = 0
                    cycle.insert(j, u)
                
                current_max = best_new_max
                improved = True
            else:
                # Local minimum reached
                break
                
        if current_max < best_score:
            best_score = current_max
            best_cycle = cycle[:]
            
    return best_cycle

if __name__ == "__main__":
    print(construct())
