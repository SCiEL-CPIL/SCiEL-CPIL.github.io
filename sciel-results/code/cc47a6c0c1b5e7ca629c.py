"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 15
    deadline = time.time() + 75
    
    # Start with K_{7,8}
    adj = [0] * n
    for i in range(7):
        for j in range(7, 15):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def get_degs():
        return [bin(m).count('1') for m in adj]
    
    def check_valid():
        degs = get_degs()
        for s in range(n):
            for t in range(s + 1, n):
                if degs[s] == degs[t]:
                    ns = adj[s] & ~(1 << t)
                    nt = adj[t] & ~(1 << s)
                    if ns == 0 or nt == 0:
                        continue
                    um = ns
                    while um:
                        lb = um & (-um)
                        u = lb.bit_length() - 1
                        if adj[u] & nt:
                            return False
                        um ^= lb
        return True
    
    def count_edges():
        return sum(bin(m).count('1') for m in adj) // 2
    
    best = count_edges()
    
    # Greedy add intra-partition edges
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    random.seed(42)
    random.shuffle(candidates)
    
    for (i, j) in candidates:
        if time.time() > deadline:
            break
        if not (adj[i] & (1 << j)):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            if check_valid():
                best += 1
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
    
    # Local search: 1-swap
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for (i, j) in all_edges:
            if time.time() > deadline:
                break
            if adj[i] & (1 << j):
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                for (k, l) in all_edges:
                    if time.time() > deadline:
                        break
                    if not (adj[k] & (1 << l)):
                        adj[k] |= (1 << l)
                        adj[l] |= (1 << k)
                        if check_valid():
                            best += 1
                            improved = True
                            break
                        else:
                            adj[k] &= ~(1 << l)
                            adj[l] &= ~(1 << k)
                if not improved:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    break
        if not improved:
            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
