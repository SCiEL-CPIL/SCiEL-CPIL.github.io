<<<<<<< SEARCH
def _get_cycle_lengths(n, edges):
    """
    Enumerate all simple cycles and return the set of their lengths.
    Uses DFS from each vertex, only visiting vertices > start to avoid duplicates.
    """
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    lengths = set()
    
    def find_cycles_from(start):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 4:
                    lengths.add(len(path))
                elif nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
                    
    for s in range(n):
        find_cycles_from(s)
        
    return lengths
=======
def _get_cycle_lengths(n, edges):
    """
    Enumerate all simple cycles and return the set of their lengths.
    Uses DFS from each vertex, only visiting vertices > start to avoid duplicates.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    lengths = set()
    
    def find_cycles_from(start):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 4:
                    lengths.add(len(path))
                elif nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
                    
    for s in range(n):
        find_cycles_from(s)
        
    return lengths
>>>>>>> REPLACE

