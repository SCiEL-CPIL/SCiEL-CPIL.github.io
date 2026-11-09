"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    best_n = n
    best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    
    def has_k5(adj, n):
        """Check if graph contains K5."""
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            if (b in adj[a] and c in adj[a] and d in adj[a] and e in adj[a] and
                c in adj[b] and d in adj[b] and e in adj[b] and
                d in adj[c] and e in adj[c] and
                e in adj[d]):
                return True
        return False
    
    def has_ind5(adj, n):
        """Check if graph contains independent set of size 5."""
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            if (b not in adj[a] and c not in adj[a] and d not in adj[a] and e not in adj[a] and
                c not in adj[b] and d not in adj[b] and e not in adj[b] and
                d not in adj[c] and e not in adj[c] and
                e not in adj[d]):
                return True
        return False
    
    def is_valid(adj, n):
        """Check if graph is (5,5)-free."""
        return not has_k5(adj, n) and not has_ind5(adj, n)
    
    # Try to add vertices
    max_vertices = 42
    
    while n < max_vertices:
        if time.time() > deadline:
            break
        
        # Try to add a new vertex with various connection patterns
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            new_n = n + 1
            new_adj = [adj[i].copy() for i in range(n)]
            new_adj.append(set())
            
            # Randomly decide connections to existing vertices
            for i in range(n):
                if random.random() < 0.5:
                    new_adj[i].add(n)
                    new_adj[n].add(i)
            
            if is_valid(new_adj, new_n):
                n = new_n
                adj = new_adj
                found = True
                break
        
        if not found:
            # Try edge flips to enable adding a vertex
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to flip
                if n > 0 and adj[0]:
                    u = random.randint(0, n - 1)
                    if adj[u]:
                        v = random.choice(list(adj[u]))
                    else:
                        continue
                else:
                    break
                
                # Flip edge
                if v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                
                if is_valid(adj, n):
                    # Try adding a vertex again
                    for add_attempt in range(100):
                        if time.time() > deadline:
                            break
                        
                        new_n = n + 1
                        new_adj = [adj[i].copy() for i in range(n)]
                        new_adj.append(set())
                        
                        for i in range(n):
                            if random.random() < 0.5:
                                new_adj[i].add(n)
                                new_adj[n].add(i)
                        
                        if is_valid(new_adj, new_n):
                            n = new_n
                            adj = new_adj
                            found = True
                            break
                    
                    if found:
                        break
                
                # Undo flip
                if v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
        
        if not found:
            break
    
    # Update best if we found a larger valid graph
    if n > best_n:
        best_n = n
        best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
