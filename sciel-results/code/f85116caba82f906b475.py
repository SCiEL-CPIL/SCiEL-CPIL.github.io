import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def check_tournament_fast(n, edge_matrix):
        """Check if every 4-set has a common dominator using bitmask approach.
        edge_matrix[i][j] = True if i -> j.
        Returns True if valid."""
        # in_neighbors[u] is bitmask of vertices that point to u
        in_neighbors = [0] * n
        
        for u in range(n):
            mask = 0
            for v in range(n):
                if v != u and edge_matrix[v][u]:
                    mask |= (1 << v)
            in_neighbors[u] = mask
            
        # Check all 4-sets
        for i in range(n):
            for j in range(i + 1, n):
                ij = in_neighbors[i] & in_neighbors[j]
                if ij == 0:
                    continue
                for k in range(j + 1, n):
                    ijk = ij & in_neighbors[k]
                    if ijk == 0:
                        continue
                    for l in range(k + 1, n):
                        if (ijk & in_neighbors[l]) == 0:
                            return False
        return True

    def make_paley_prime_matrix(p):
        """Create edge matrix for Paley tournament on prime p vertices.
        p must be prime with p ≡ 3 (mod 4)."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # edge_matrix[i][j] = True if i -> j
        edge_matrix = [[False] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                edge_matrix[i][j] = diff in squares
        return edge_matrix

    def build_orientation_from_matrix(edge_matrix, n):
        """Build orientation list from edge matrix."""
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if edge_matrix[i][j]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Try primes p ≡ 3 (mod 4) in increasing order
    # Smallest such primes: 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83
    candidate_primes = []
    for p in range(5, 91):
        if p % 4 != 3:
            continue
        is_prime = True
        for i in range(2, int(p ** 0.5) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            candidate_primes.append(p)

    # Try each candidate prime
    for p in candidate_primes:
        if time.time() - start_time > deadline - 5:
            break
        
        edge_matrix = make_paley_prime_matrix(p)
        
        if check_tournament_fast(p, edge_matrix):
            orientation = build_orientation_from_matrix(edge_matrix, p)
            return {'n': p, 'orientation': orientation}

    # Fallback: return Paley(67) which is known to work
    p = 67
    edge_matrix = make_paley_prime_matrix(p)
    orientation = build_orientation_from_matrix(edge_matrix, p)
    return {'n': p, 'orientation': orientation}
