"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20
    best_edges = [[0, v, 0] for v in range(1, 20)]  # star fallback
    best_count = 19
    
    # Strategy: try dense bipartite graphs K_{10,10} with proper edge-colourings
    # A proper edge-colouring of K_{10,10} uses 10 colours.
    # Check: every 4-cycle must be rainbow.
    
    def check_and_count(adj, colours, n):
        """Check all 4-cycles are rainbow. Return (valid, edge_count)."""
        edge_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    edge_count += 1
        
        # Check all 4-cycles
        for a, b, c, d in combinations(range(n), 4):
            # 4-cycle: a-b-c-d-a
            if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]:
                cols = [colours[(a,b)], colours[(b,c)], colours[(c,d)], colours[(d,a)]]
                if len(set(cols)) != 4:
                    return False, edge_count
            # 4-cycle: a-b-d-c-a
            if b in adj[a] and d in adj[b] and c in adj[d] and a in adj[c]:
                cols = [colours[(a,b)], colours[(b,d)], colours[(d,c)], colours[(c,a)]]
                if len(set(cols)) != 4:
                    return False, edge_count
            # 4-cycle: a-c-b-d-a
            if c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]:
                cols = [colours[(a,c)], colours[(c,b)], colours[(b,d)], colours[(d,a)]]
                if len(set(cols)) != 4:
                    return False, edge_count
        return True, edge_count
    
    # Try K_{10,10} with various proper edge-colourings
    # Proper edge-colouring via 1-factorization: colour(i,j) = (i - j) % 10
    # But we need to check 4-cycle rainbow condition.
    
    left = list(range(10))
    right = list(range(10, 20))
    
    # Try multiple 1-factorizations (permutations of colour assignment)
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random 1-factorization: shuffle the colour assignment
        perm = list(range(10))
        random.shuffle(perm)
        
        adj = {i: set() for i in range(n)}
        colours = {}
        
        for i in left:
            for j in right:
                c = perm[(i - (j - 10)) % 10]
                adj[i].add(j)
                adj[j].add(i)
                colours[(i, j)] = c
                colours[(j, i)] = c
        
        valid, count = check_and_count(adj, colours, n)
        if valid and count > best_count:
            best_count = count
            best_edges = [[i, j, colours[(i,j)]] for i in range(n) for j in adj[i] if i < j]
    
    # Also try adding cross edges between left and right beyond K_{10,10}
    # Actually K_{10,10} IS the complete bipartite, so 100 edges max for this structure.
    
    # Try other bipartitions
    for split in [8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        left = list(range(split))
        right = list(range(split, n))
        m_left = len(left)
        m_right = len(right)
        m = min(m_left, m_right)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            perm = list(range(20))
            random.shuffle(perm)
            
            adj = {i: set() for i in range(n)}
            colours = {}
            
            for i in left:
                for j in right:
                    # Proper edge-colouring: use (i_idx - j_idx) mod min(m_left,m_right)
                    ii = left.index(i)
                    jj = right.index(j)
                    c = perm[(ii - jj) % m]
                    adj[i].add(j)
                    adj[j].add(i)
                    colours[(i, j)] = c
                    colours[(j, i)] = c
            
            valid, count = check_and_count(adj, colours, n)
            if valid and count > best_count:
                best_count = count
                best_edges = [[i, j, colours[(i,j)]] for i in range(n) for j in adj[i] if i < j]
    
    return best_edges
# EVOLVE-BLOCK-END
