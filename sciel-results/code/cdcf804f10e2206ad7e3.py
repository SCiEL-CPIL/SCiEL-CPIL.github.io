import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    total_edges = len(edges)
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def popcount(x):
        return bin(x).count('1')
    
    def bits_to_list(x):
        result = []
        while x:
            lsb = x & -x
            result.append(lsb.bit_length() - 1)
            x ^= lsb
        return result
    
    def find_biclique(uncovered, start_u, start_v):
        """Find a large biclique containing edge (start_u, start_v) in uncovered subgraph."""
        L = {start_u}
        R = {start_v}
        # Common neighbors that can be added
        # For L: candidate u' must connect to all in R
        # For R: candidate v' must connect to all in L
        
        changed = True
        while changed:
            changed = False
            # Try to add to L
            best_u_prime = -1
            best_score = 0
            for u_prime in range(n):
                if u_prime in L:
                    continue
                # Check if u_prime connects to all in R via uncovered edges
                valid = True
                for v in R:
                    if not (uncovered[u_prime] & (1 << v)):
                        valid = False
                        break
                if valid:
                    # Score: total uncovered degree
                    score = popcount(uncovered[u_prime])
                    if score > best_score:
                        best_score = score
                        best_u_prime = u_prime
            if best_u_prime != -1:
                L.add(best_u_prime)
                changed = True
                continue
            
            # Try to add to R
            best_v_prime = -1
            best_score = 0
            for v_prime in range(n):
                if v_prime in R:
                    continue
                valid = True
                for u in L:
                    if not (uncovered[u] & (1 << v_prime)):
                        valid = False
                        break
                if valid:
                    score = popcount(uncovered[v_prime])
                    if score > best_score:
                        best_score = score
                        best_v_prime = v_prime
            if best_v_prime != -1:
                R.add(best_v_prime)
                changed = True
        
        return L, R
    
    def greedy_cover(seed, strategy=0):
        """Greedy biclique cover with different strategies."""
        random.seed(seed)
        uncovered = [0] * n
        for u, v in edges:
            uncovered[u] |= (1 << v)
            uncovered[v] |= (1 << u)
        
        bicliques = []
        remaining = total_edges
        
        while remaining > 0:
            # Find an uncovered edge
            if strategy == 0:
                # Pick vertex with highest uncovered degree
                best_u = -1
                best_deg = -1
                for u in range(n):
                    deg = popcount(uncovered[u])
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
            elif strategy == 1:
                # Random vertex with uncovered edges
                candidates = [u for u in range(n) if uncovered[u]]
                if not candidates:
                    break
                best_u = random.choice(candidates)
            elif strategy == 2:
                # Pick vertex with highest degree, then random neighbor
                best_u = -1
                best_deg = -1
                for u in range(n):
                    deg = popcount(uncovered[u])
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
            else:
                best_u = -1
                best_deg = -1
                for u in range(n):
                    deg = popcount(uncovered[u])
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
            
            if best_u == -1 or not uncovered[best_u]:
                break
            
            # Pick a neighbor
            v_mask = uncovered[best_u]
            neighbors = bits_to_list(v_mask)
            if not neighbors:
                break
            
            if strategy == 0:
                # Pick neighbor with highest degree
                best_v = max(neighbors, key=lambda x: popcount(uncovered[x]))
            elif strategy == 1:
                best_v = random.choice(neighbors)
            else:
                best_v = random.choice(neighbors)
            
            # Find biclique
            L, R = find_biclique(uncovered, best_u, best_v)
            
            bicliques.append((L, R))
            
            # Remove covered edges
            for u in L:
                for v in R:
                    if uncovered[u] & (1 << v):
                        uncovered[u] &= ~(1 << v)
                        uncovered[v] &= ~(1 << u)
                        remaining -= 1
        
        return bicliques
    
    def validate(bicliques):
        """Validate that bicliques form a valid partition."""
        # Check all cross pairs are edges
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            L_set = set(L)
            R_set = set(R)
            if L_set & R_set:
                return False
            for u in L:
                for v in R:
                    key = (min(u,v), max(u,v))
                    if key not in edge_set:
                        return False
                    if key in covered:
                        return False
                    covered.add(key)
        
        if covered != edge_set:
            return False
        return True
    
    def try_improve(bicliques):
        """Try to merge or improve bicliques."""
        # For now, just return as is
        return bicliques
    
    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Try multiple strategies and seeds
    seed = 0
    while time.time() < deadline:
        for strategy in range(4):
            if time.time() >= deadline:
                break
            bicliques = greedy_cover(seed, strategy)
            if validate(bicliques):
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_bicliques = bicliques
                    if best_count <= 4:
                        break
        seed += 1
        if best_count <= 4:
            break
    
    if best_bicliques is None:
        # Fallback: each edge as its own biclique
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    # Convert to required format
    result = []
    for L, R in best_bicliques:
        result.append([list(map(int, L)), list(map(int, R))])
    
    return result
