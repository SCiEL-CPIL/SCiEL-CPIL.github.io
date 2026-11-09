import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin

    def is_valid_5_5(n, adj):
        """
        Checks if the graph defined by n vertices and adjacency bitmasks 'adj'
        contains no K5 and no independent set of size 5.
        Returns True if valid (no K5, no indep5), False otherwise.
        """
        # Precompute list of vertices for combinations
        vertices = range(n)
        
        # Check for K5
        for combo in itertools.combinations(vertices, 5):
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if not (adj[u] & (1 << v)):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False

        # Check for Independent Set of size 5
        for combo in itertools.combinations(vertices, 5):
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if (adj[u] & (1 << v)):
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False

        return True

    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Try to find a valid circulant graph for n > 37
    # We try n from 42 down to 38
    for n in range(42, 37, -1):
        if time.time() > deadline - 30:
            break
        
        # Random search for circulant graph
        # Connection set S is a subset of {1, ..., n//2}
        max_dist = n // 2
        possible_dists = list(range(1, max_dist + 1))
        
        # We will sample random subsets
        # To make it more effective, we can try different sizes
        for _ in range(200): # Max 200 trials per n
            if time.time() > deadline:
                break
            
            # Random subset
            # Bias towards certain sizes? Just random for now.
            # Size around n/4 to n/3 is typical for Ramsey graphs
            size = random.randint(n//4, n//3 + 1)
            S = set(random.sample(possible_dists, min(size, len(possible_dists))))
            
            # Build adjacency list
            adj = [0] * n
            for i in range(n):
                for d in S:
                    j1 = (i + d) % n
                    j2 = (i - d) % n
                    adj[i] |= (1 << j1)
                    adj[i] |= (1 << j2)
            
            # Check validity
            if is_valid_5_5(n, adj):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges

    # Fallback: Paley graph on 37 vertices
    return get_paley_37()
