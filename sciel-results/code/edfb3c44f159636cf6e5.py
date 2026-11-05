import itertools

def construct():
    # List of candidate primes p < 67, p >= 5
    candidates = [7, 11, 19, 23, 31, 43, 47, 59]
    
    def get_paley_orientation(p):
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # i -> j if (j - i) % p in squares
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def is_schutte(p):
        # Check if Paley tournament on p vertices is a Schütte tournament for k=4
        # Every 4 vertices must have a common dominator (vertex outside the set with arcs to all 4)
        squares = {x * x % p for x in range(1, p)}
        
        # Precompute out-neighbors for each vertex
        # out[u] is a set of v such that u -> v
        out = [set() for _ in range(p)]
        for u in range(p):
            for v in range(p):
                if u == v: continue
                if (v - u) % p in squares:
                    out[u].add(v)
        
        # Check all 4-subsets
        # A 4-subset S is bad if there is no v not in S such that v -> u for all u in S
        # This is equivalent to: intersection of out[u] for u in S, excluding S, is empty.
        for S in itertools.combinations(range(p), 4):
            # Intersect out-neighbors of all vertices in S
            common_out = out[S[0]].intersection(out[S[1]], out[S[2]], out[S[3]])
            # Remove vertices in S
            common_out -= set(S)
            if not common_out:
                return False
        return True

    # Test candidates
    for p in candidates:
        if is_schutte(p):
            return {'n': p, 'orientation': get_paley_orientation(p)}
            
    # Fallback to known valid solution
    p = 67
    return {'n': p, 'orientation': get_paley_orientation(p)}
