import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 20
    
    def init_graph():
        # Start with complete graph K20
        N = [0] * n
        for i in range(n):
            # Set all bits except self
            mask = (1 << n) - 1
            mask &= ~(1 << i)
            N[i] = mask
        return N

    def is_valid(N):
        # Check if any 3 vertices have >= 3 common neighbors
        # Returns (True, None) if valid, (False, (u, v, w)) if violation found
        for u in range(n):
            for v in range(u + 1, n):
                C = N[u] & N[v]
                if C == 0:
                    continue
                # Iterate through bits in C to find w
                temp = C
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    
                    # Check if N[u] & N[v] & N[w] has >= 3 bits
                    # Note: w is in C, so w is in N[u] and N[v].
                    # Intersection is C & N[w]
                    if (C & N[w]).bit_count() >= 3:
                        return False, (u, v, w)
        return True, None

    best_edges = 0
    best_N = None
    
    # Run until deadline
    while time.time() < deadline:
        N = init_graph()
        
        # Greedy deletion loop
        while True:
            valid, viol = is_valid(N)
            if valid:
                # Count edges
                edge_count = sum(N[i].bit_count() for i in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_N = N.copy()
                break
            
            u, v, w = viol
            # Violation: N[u] & N[v] & N[w] has >= 3 common neighbors
            C = N[u] & N[v] & N[w]
            
            # Pick a random common neighbor y to disconnect
            # Get indices of bits in C
            ys = []
            temp = C
            while temp:
                lsb = temp & -temp
                y = lsb.bit_length() - 1
                ys.append(y)
                temp ^= lsb
            
            y = random.choice(ys)
            
            # Pick a random vertex x from the triple {u, v, w} to disconnect from y
            x = random.choice([u, v, w])
            
            # Remove edge (x, y)
            N[x] &= ~(1 << y)
            N[y] &= ~(1 << x)
            
    # Construct edge list from best_N
    edges = []
    if best_N:
        for i in range(n):
            for j in range(i + 1, n):
                if best_N[i] & (1 << j):
                    edges.append([i, j])
    else:
        # Fallback to seed if something went wrong
        edges = [[a, b] for a in range(2) for b in range(2, 20)]
        
    return edges
