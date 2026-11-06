import time
import itertools

def is_k5_free(n, adj):
    # adj is a list of sets or bitmasks
    # Check for K5
    for comb in itertools.combinations(range(n), 5):
        is_clique = True
        for i in range(5):
            for j in range(i+1, 5):
                u, v = comb[i], comb[j]
                if v not in adj[u]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_n = 37
    best_edges = None
    
    # Generate Paley graph for q=37 as fallback
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        n = q
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append((i, j))
        return n, adj, edges

    # Check q=37
    n, adj, edges = get_paley(37)
    if is_k5_free(n, adj):
        best_n = n
        best_edges = edges

    # Try other q
    candidates = [41, 53, 59, 61, 73, 89, 97]
    for q in candidates:
        if time.time() > deadline:
            break
        n, adj, edges = get_paley(q)
        if is_k5_free(n, adj):
            if n > best_n:
                best_n = n
                best_edges = edges

    return best_n, best_edges
