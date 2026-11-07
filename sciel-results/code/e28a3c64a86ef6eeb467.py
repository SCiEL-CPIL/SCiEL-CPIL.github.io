import time

P = {'n': 18, 'edge_order': [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [6, 14, 17], [11, 12, 16], [3, 10, 17], [10, 12, 14], [3, 9, 13], [0, 7, 13], [3, 14, 16], [1, 5, 8], [0, 9, 12], [9, 13, 17], [4, 11, 17], [4, 8, 13], [3, 14, 17], [3, 11, 15], [11, 13, 17], [8, 11, 12], [4, 9, 13], [2, 7, 9], [3, 6, 14], [0, 3, 5], [5, 7, 16], [2, 9, 16], [6, 7, 9], [1, 7, 8], [3, 10, 13], [3, 10, 14], [1, 10, 15], [2, 4, 14], [4, 11, 14], [6, 7, 10], [5, 6, 12], [3, 13, 17], [0, 5, 7], [5, 7, 11], [0, 2, 11], [9, 15, 16], [0, 2, 14], [5, 15, 17], [0, 3, 7], [10, 12, 15], [7, 9, 11], [4, 7, 17], [4, 11, 16], [3, 6, 17], [7, 11, 14], [1, 3, 8], [1, 10, 16], [3, 9, 16], [7, 8, 11], [7, 10, 14], [4, 10, 14], [3, 14, 15], [4, 14, 17], [4, 16, 17], [9, 12, 15], [0, 2, 4], [7, 9, 15], [10, 12, 17], [7, 8, 13], [8, 11, 15], [8, 16, 17], [5, 11, 13], [1, 11, 14], [0, 6, 10], [0, 8, 14], [5, 6, 16], [0, 6, 17], [1, 16, 17], [7, 11, 16], [2, 11, 16], [0, 1, 13], [6, 7, 14], [4, 7, 11], [6, 11, 14], [1, 7, 14], [1, 4, 8], [5, 9, 15], [8, 9, 15], [3, 11, 13], [2, 11, 14], [3, 12, 13], [2, 4, 8], [12, 14, 15], [1, 12, 17], [1, 14, 17], [6, 15, 16], [0, 11, 13], [2, 5, 15], [11, 12, 17], [5, 15, 16], [2, 6, 15], [5, 6, 15], [5, 8, 15], [0, 7, 9], [1, 4, 14], [9, 14, 15], [2, 4, 10], [3, 7, 15], [5, 7, 17], [1, 10, 14], [6, 8, 17], [1, 2, 15], [1, 14, 15], [2, 5, 17], [7, 9, 16], [10, 13, 17], [5, 9, 14], [0, 12, 13], [0, 13, 14], [7, 10, 12], [2, 11, 15], [4, 5, 10], [10, 16, 17], [4, 14, 16], [1, 8, 15], [5, 6, 14], [7, 11, 15], [3, 11, 12], [7, 9, 17], [4, 7, 14], [1, 5, 15], [3, 7, 12], [4, 5, 12], [3, 4, 15], [1, 6, 10], [1, 8, 10], [6, 10, 13], [9, 11, 17], [10, 14, 15], [2, 3, 5], [0, 5, 6], [4, 11, 12], [2, 6, 16], [2, 8, 10], [2, 3, 11], [9, 10, 11], [2, 9, 17], [0, 12, 16], [0, 6, 13], [1, 9, 13], [5, 6, 9], [4, 5, 17], [2, 7, 10], [2, 3, 7], [1, 7, 11], [3, 7, 17], [0, 2, 7], [3, 5, 10], [4, 7, 15], [1, 2, 11], [6, 8, 10], [2, 4, 6], [0, 2, 13], [2, 3, 17], [3, 7, 16], [3, 4, 10], [1, 6, 7], [6, 10, 12], [4, 6, 7], [7, 10, 17], [8, 13, 17], [1, 12, 15], [0, 8, 15], [2, 5, 7], [1, 3, 15], [2, 4, 11], [4, 6, 8], [9, 14, 16], [0, 13, 16], [0, 9, 16], [8, 10, 14], [3, 5, 12], [2, 9, 13], [2, 7, 12], [0, 1, 11], [1, 7, 13], [1, 4, 7], [0, 9, 17], [0, 1, 14], [8, 13, 14], [0, 2, 5], [1, 10, 12], [8, 12, 13], [1, 14, 16], [3, 8, 12], [5, 7, 14], [1, 5, 9], [5, 13, 16], [0, 1, 15], [0, 16, 17], [3, 5, 6], [5, 8, 11], [0, 3, 13], [0, 12, 14], [5, 6, 8], [5, 7, 12], [4, 5, 8], [0, 10, 11], [1, 8, 11], [1, 13, 16], [3, 12, 16], [3, 6, 8], [0, 2, 17], [9, 10, 12], [2, 5, 8], [1, 12, 14], [8, 9, 13], [0, 7, 8], [0, 14, 17], [4, 10, 13], [9, 12, 13], [5, 11, 15], [13, 14, 15], [1, 9, 10], [2, 7, 15], [6, 9, 10], [0, 12, 15], [12, 13, 16], [0, 9, 14], [6, 13, 15], [1, 4, 10], [4, 7, 16], [12, 15, 17], [2, 7, 14], [0, 4, 5], [3, 16, 17], [13, 15, 16], [5, 7, 10], [6, 7, 16], [6, 15, 17], [6, 12, 13], [1, 13, 15], [0, 4, 8], [7, 13, 15], [4, 9, 11], [6, 7, 13], [0, 8, 16], [5, 6, 11], [2, 4, 17], [3, 13, 14], [10, 11, 13], [2, 11, 12], [6, 7, 17], [4, 13, 15], [3, 5, 13], [0, 9, 15], [3, 13, 16], [4, 12, 15], [12, 16, 17], [2, 6, 12], [7, 8, 12], [0, 7, 17], [2, 10, 12], [8, 9, 16], [8, 10, 13], [6, 9, 16], [6, 13, 17], [1, 3, 13], [6, 11, 12], [6, 13, 14], [2, 4, 9], [0, 10, 15], [0, 14, 16], [1, 9, 17], [4, 8, 10], [3, 6, 13], [4, 6, 17], [1, 2, 8], [2, 10, 17], [2, 8, 16], [0, 7, 16], [0, 10, 17], [2, 14, 15], [6, 7, 11], [10, 11, 16], [6, 9, 11], [1, 8, 17], [1, 7, 9], [0, 6, 9], [10, 14, 17], [2, 5, 10], [4, 8, 9], [7, 8, 15], [2, 12, 17], [9, 10, 14], [8, 12, 17], [6, 7, 12], [5, 8, 14], [6, 9, 14], [5, 12, 14], [3, 15, 17], [10, 13, 16], [8, 14, 15], [7, 11, 17], [5, 10, 13], [5, 11, 14], [6, 12, 15], [2, 10, 15], [0, 5, 15], [12, 14, 17], [11, 13, 16], [2, 13, 16], [2, 9, 15], [0, 15, 17], [0, 6, 11], [2, 8, 13], [0, 1, 4], [4, 12, 13], [5, 9, 12], [0, 9, 10], [1, 9, 11], [7, 15, 16], [0, 1, 8], [5, 7, 9], [0, 1, 9], [11, 12, 14], [1, 8, 14], [6, 8, 12], [5, 8, 17], [3, 8, 15], [3, 5, 17], [2, 5, 11], [0, 3, 12], [5, 14, 16], [7, 14, 17], [6, 9, 12], [9, 10, 15], [7, 9, 13], [9, 11, 16], [3, 7, 11], [3, 9, 12], [3, 4, 7], [1, 5, 17], [0, 1, 3], [5, 10, 12], [2, 3, 8], [7, 13, 14], [0, 7, 12], [9, 13, 15], [3, 11, 14], [0, 4, 16], [12, 13, 15], [7, 9, 14], [2, 10, 16], [7, 9, 10], [3, 10, 12], [14, 15, 16], [5, 12, 16], [1, 3, 10], [4, 10, 15], [5, 6, 13], [1, 7, 10], [6, 9, 13], [2, 9, 11], [1, 7, 15], [1, 7, 16], [3, 10, 16], [1, 6, 13], [0, 9, 13], [5, 8, 13], [11, 13, 15], [2, 4, 15], [10, 11, 14], [13, 16, 17], [3, 5, 7], [0, 8, 13], [3, 8, 17], [2, 5, 6], [3, 4, 6], [0, 1, 10], [3, 8, 10], [0, 3, 8], [2, 13, 17], [0, 4, 15], [2, 7, 13], [3, 12, 17], [8, 10, 11], [4, 8, 12], [10, 12, 13], [11, 12, 15], [0, 4, 10], [8, 10, 17], [4, 5, 11], [0, 10, 16], [0, 6, 7], [3, 10, 11], [1, 15, 17], [3, 8, 13], [3, 6, 9], [4, 15, 17], [2, 4, 13], [4, 8, 11], [1, 10, 17], [0, 2, 12], [0, 7, 10], [0, 1, 16], [0, 7, 14], [4, 5, 6], [4, 5, 9], [2, 6, 10], [2, 6, 17], [0, 7, 15], [4, 5, 14], [3, 7, 10], [2, 16, 17], [5, 8, 12], [6, 11, 16], [6, 10, 15], [2, 15, 17], [4, 6, 14], [0, 10, 12], [10, 15, 17], [1, 6, 16], [0, 12, 17], [6, 11, 13], [1, 3, 9], [7, 12, 16], [3, 5, 14], [6, 10, 11], [3, 11, 16], [1, 4, 12], [7, 15, 17], [4, 9, 17], [7, 8, 17], [1, 8, 13], [3, 6, 12], [1, 13, 17], [2, 9, 10], [7, 9, 12], [1, 8, 9], [2, 8, 17], [0, 11, 14], [3, 6, 11], [2, 5, 12], [11, 15, 16], [1, 9, 12], [2, 7, 16], [5, 8, 10], [2, 6, 14], [9, 13, 14], [5, 9, 10], [2, 12, 14], [0, 8, 11], [8, 12, 16], [3, 4, 8], [7, 11, 13], [5, 14, 15], [2, 4, 16], [1, 2, 10], [11, 14, 15], [6, 8, 15], [10, 11, 15], [6, 8, 16], [11, 12, 13], [8, 11, 17], [11, 15, 17], [2, 3, 13], [1, 5, 11], [0, 5, 16], [4, 6, 15], [14, 16, 17], [0, 9, 11], [0, 3, 10], [0, 2, 3], [3, 10, 15], [10, 12, 16], [2, 11, 13], [4, 5, 7], [10, 15, 16], [8, 14, 17], [1, 13, 14], [2, 6, 8], [1, 2, 17], [2, 4, 5], [9, 13, 16], [4, 10, 12], [7, 14, 16], [5, 7, 13], [3, 4, 16], [2, 12, 15], [0, 3, 4], [6, 7, 15], [8, 11, 14], [5, 9, 13], [1, 2, 9], [0, 8, 12], [1, 5, 16], [0, 5, 14], [8, 10, 16], [1, 10, 13], [10, 11, 12], [4, 15, 16], [5, 13, 15], [1, 10, 11], [13, 14, 16], [2, 10, 14], [1, 5, 7], [6, 14, 16], [4, 6, 11], [2, 3, 12], [7, 14, 15], [7, 10, 16], [4, 12, 17], [0, 8, 10], [0, 7, 11], [1, 9, 14], [8, 13, 16], [4, 13, 16], [1, 8, 12], [1, 6, 14], [9, 11, 14], [8, 10, 12], [1, 3, 5], [0, 3, 14], [0, 4, 7], [2, 13, 14], [5, 11, 17], [1, 2, 16], [3, 7, 8], [2, 11, 17], [5, 13, 14], [3, 5, 11], [4, 8, 14], [1, 3, 17], [0, 14, 15], [5, 8, 9], [0, 15, 16], [0, 11, 12], [0, 2, 16], [2, 3, 9], [2, 5, 13], [8, 12, 15], [9, 10, 13], [7, 10, 13], [0, 6, 12], [3, 11, 17], [1, 5, 13], [3, 7, 9], [0, 1, 2], [1, 12, 13], [4, 9, 10], [1, 4, 6], [8, 9, 11], [3, 5, 15], [2, 14, 17], [2, 9, 14], [2, 8, 11], [5, 12, 13], [7, 10, 11], [9, 12, 16], [5, 12, 17], [0, 5, 10], [10, 14, 16], [5, 10, 15], [8, 9, 17], [1, 6, 15], [1, 8, 16], [6, 13, 16], [4, 5, 15], [8, 13, 15], [6, 10, 14], [3, 7, 13], [3, 4, 5], [0, 2, 9], [5, 6, 7], [8, 15, 17], [0, 11, 16], [5, 7, 15], [7, 12, 15], [6, 8, 9], [0, 1, 12], [6, 8, 11], [0, 1, 7], [0, 5, 8], [0, 13, 15], [0, 3, 11], [0, 4, 17], [9, 15, 17], [5, 11, 16], [9, 16, 17], [4, 11, 15], [8, 12, 14], [3, 5, 16], [3, 5, 8], [0, 6, 16], [2, 3, 14], [4, 6, 12], [4, 12, 16], [6, 8, 13], [5, 9, 17], [5, 9, 11], [3, 7, 14], [8, 9, 12], [0, 5, 17], [4, 7, 10], [3, 9, 14], [3, 9, 17], [1, 2, 4], [11, 14, 16], [0, 2, 10], [1, 3, 6], [2, 8, 15], [5, 8, 16], [3, 4, 9], [2, 8, 14], [2, 5, 9], [1, 4, 5], [4, 6, 13], [9, 12, 17], [7, 12, 17], [3, 6, 16], [10, 13, 14], [5, 10, 17], [6, 11, 17], [5, 12, 15], [4, 12, 14], [2, 7, 17], [3, 12, 14], [0, 3, 15], [1, 4, 16], [6, 11, 15], [0, 4, 9], [0, 10, 14], [0, 10, 13], [4, 9, 16], [0, 6, 8], [0, 4, 14], [4, 9, 14], [2, 14, 16], [13, 14, 17], [0, 3, 9], [2, 8, 12], [7, 16, 17], [6, 14, 15], [3, 13, 15], [4, 10, 16], [9, 11, 15], [4, 7, 9], [4, 10, 11], [1, 2, 5], [6, 7, 8], [0, 3, 17], [3, 9, 15], [0, 13, 17], [3, 12, 15], [5, 6, 17], [4, 5, 13], [6, 10, 17], [1, 5, 10], [1, 2, 6], [6, 12, 14], [11, 13, 14], [1, 11, 17], [0, 6, 14], [0, 2, 8], [1, 2, 7], [1, 6, 8], [1, 3, 12], [3, 4, 11], [7, 12, 13], [12, 13, 14], [4, 7, 13], [2, 5, 14], [1, 2, 3], [7, 8, 10], [0, 4, 12], [0, 4, 6], [2, 3, 4], [8, 14, 16], [1, 3, 16], [1, 15, 16], [3, 15, 16], [0, 5, 13], [1, 9, 16], [6, 9, 15], [5, 9, 16], [1, 5, 14], [7, 13, 16], [1, 4, 15], [10, 11, 17], [1, 3, 4], [2, 12, 16], [0, 11, 17], [4, 9, 12], [5, 13, 17], [2, 6, 13], [4, 9, 15], [5, 7, 8], [8, 11, 13], [1, 9, 15], [9, 10, 16], [0, 11, 15], [0, 2, 6], [0, 1, 17], [1, 6, 17], [9, 11, 13], [2, 9, 12], [4, 7, 8], [1, 2, 12], [1, 11, 16], [4, 11, 13], [1, 11, 12], [4, 8, 15], [2, 3, 15], [2, 15, 16], [2, 10, 13], [0, 5, 12], [13, 15, 17], [3, 4, 13], [5, 16, 17], [9, 11, 12], [2, 10, 11], [3, 4, 14], [6, 10, 16], [8, 10, 15], [4, 5, 16], [4, 6, 10], [2, 3, 6], [8, 9, 14], [9, 12, 14], [5, 10, 11], [12, 14, 16], [0, 8, 9], [6, 16, 17], [4, 6, 9], [15, 16, 17], [7, 12, 14], [3, 5, 9], [1, 7, 12], [0, 8, 17], [0, 5, 11], [12, 13, 17], [6, 12, 16], [5, 10, 14], [1, 3, 11], [3, 9, 11], [7, 8, 16], [1, 4, 13], [1, 6, 9], [6, 9, 17], [8, 15, 16], [1, 3, 7], [0, 1, 5], [6, 12, 17], [3, 4, 17], [8, 11, 16], [11, 16, 17], [3, 8, 9], [7, 13, 17], [1, 6, 12], [1, 2, 13], [4, 13, 17], [0, 4, 11], [2, 6, 9], [2, 6, 11], [5, 6, 10], [7, 8, 9], [2, 3, 10], [0, 4, 13], [3, 6, 10], [6, 8, 14], [2, 7, 11], [4, 7, 12], [10, 13, 15], [1, 5, 12], [5, 14, 17], [2, 4, 7], [1, 4, 17], [4, 14, 15], [3, 8, 16], [12, 15, 16], [7, 8, 14], [2, 4, 12], [9, 14, 17], [3, 8, 14], [1, 4, 11], [0, 5, 9], [4, 8, 17], [1, 11, 15], [1, 4, 9], [0, 3, 16], [4, 8, 16], [0, 3, 6], [3, 8, 11], [4, 10, 17], [1, 7, 17], [8, 9, 10], [3, 4, 12], [2, 8, 9], [5, 11, 12], [1, 12, 16], [3, 9, 10], [2, 7, 8], [1, 3, 14], [4, 6, 16], [2, 5, 16], [1, 6, 11], [2, 13, 15], [5, 10, 16], [14, 15, 17], [0, 2, 15], [3, 6, 15], [2, 12, 13], [11, 14, 17], [2, 6, 7], [1, 2, 14], [1, 5, 6], [7, 10, 15], [7, 11, 12], [0, 6, 15], [3, 6, 7], [1, 11, 13]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85

    edge_order = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1

    # Map each unique triple to its earliest 1-based index
    triple_to_index = {}
    for i, tri in enumerate(edge_order):
        key = frozenset(tri)
        if key not in triple_to_index:
            triple_to_index[key] = i + 1

    # Build list of unique triples with masks
    unique_triples = []
    for tri, idx in triple_to_index.items():
        mask = 0
        for v in tri:
            mask |= (1 << v)
        unique_triples.append((idx, mask, list(tri)))

    unique_triples.sort(key=lambda x: x[0])
    M = len(unique_triples)

    # Build index: for each vertex, which unique triple indices contain it
    vertex_to_triples = [[] for _ in range(n)]
    for i, (idx, mask, tri) in enumerate(unique_triples):
        for v in tri:
            vertex_to_triples[v].append(i)

    def can_match(K, deadline):
        """Check if there's a perfect matching using only unique triples[0..K-1].
        Returns the matching as a list of triple indices, or None."""
        # Filter to first K unique triples
        allowed = set(range(K))
        
        # For each vertex, get allowed triple indices
        v_allowed = []
        for v in range(n):
            v_allowed.append([t for t in vertex_to_triples[v] if t in allowed])
        
        # If any vertex has no allowed triples, fail
        for v in range(n):
            if not v_allowed[v]:
                return None
        
        # Backtracking with MRV
        used_triples = set()
        assignment = [None] * n  # assignment[v] = triple index
        
        def backtrack(unassigned_mask):
            if time.time() > deadline:
                return False
            if unassigned_mask == 0:
                return True
            
            # MRV: pick unassigned vertex with fewest options
            best_v = -1
            best_count = float('inf')
            mv = unassigned_mask
            while mv:
                v = (mv & -mv).bit_length() - 1
                mv &= mv - 1
                cnt = 0
                for t in v_allowed[v]:
                    if t not in used_triples:
                        cnt += 1
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    if cnt == 0:
                        return False
                    if cnt == 1:
                        break
            
            if best_v == -1:
                return False
            
            # Try each allowed triple for best_v
            for t in v_allowed[best_v]:
                if t in used_triples:
                    continue
                idx, mask, tri = unique_triples[t]
                # Check all vertices in this triple are unassigned
                if mask & unassigned_mask != mask:
                    continue
                
                # Assign
                used_triples.add(t)
                new_mask = unassigned_mask & ~mask
                for v in tri:
                    assignment[v] = t
                
                if backtrack(new_mask):
                    return True
                
                # Undo
                used_triples.discard(t)
                for v in tri:
                    assignment[v] = None
            
            return False
        
        if backtrack(full_mask):
            # Extract matching
            result = []
            for t in sorted(set(assignment)):
                result.append(unique_triples[t][2])
            return result
        return None

    # Binary search on K (prefix size in unique triple space)
    # But the score is the max 1-based index in edge_order, not the unique triple rank
    # We need to find the minimum max_index such that a perfect matching exists
    # using only triples with edge_order index <= max_index
    
    # Collect all unique triples sorted by their edge_order index
    # unique_triples is already sorted by idx
    
    # Binary search on the edge_order index
    lo = 1
    hi = M  # max unique triple index
    
    best_result = None
    best_score = float('inf')
    
    # First, try to find a good upper bound quickly with a few greedy attempts
    # Greedy: try to build matching using earliest triples
    def greedy_match(max_idx, deadline):
        """Try to build a matching using only triples with index <= max_idx."""
        # Get all unique triples with index <= max_idx
        allowed_triples = []
        for i, (idx, mask, tri) in enumerate(unique_triples):
            if idx <= max_idx:
                allowed_triples.append((mask, tri))
        
        if len(allowed_triples) < 6:
            return None
        
        # Backtracking
        used = set()
        
        def bt(remaining, depth):
            if time.time() > deadline:
                return None
            if depth == 6:
                return []
            if remaining == 0:
                return []
            
            # Pick vertex with MRV
            best_v = -1
            best_opts = None
            mv = remaining
            while mv:
                v = (mv & -mv).bit_length() - 1
                mv &= mv - 1
                opts = []
                for mi, (mask, tri) in enumerate(allowed_triples):
                    if mi in used:
                        continue
                    if mask & remaining == mask:
                        opts.append(mi)
                if best_opts is None or len(opts) < len(best_opts):
                    best_v = v
                    best_opts = opts
                    if len(opts) == 0:
                        return None
                    if len(opts) == 1:
                        break
            
            if best_v == -1 or not best_opts:
                return None
            
            for mi in best_opts:
                mask, tri = allowed_triples[mi]
                used.add(mi)
                result = bt(remaining & ~mask, depth + 1)
                if result is not None:
                    return [tri] + result
                used.discard(mi)
            
            return None
        
        return bt(full_mask, 0)
    
    # Quick greedy to find initial upper bound
    for trial_max in [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500]:
        if time.time() > deadline - 5:
            break
        res = greedy_match(trial_max, deadline - 2)
        if res is not None:
            best_result = res
            best_score = trial_max
            break
    
    if best_result is None:
        # Fallback to full DP approach
        reachable = {0}
        takes = [{} for _ in range(M + 1)]
        found_k = -1
        
        for k in range(1, M + 1):
            if time.time() > deadline - 5:
                break
            idx, mask, tri_list = unique_triples[k-1]
            new_reachable = set(reachable)
            new_takes = {}
            for pmask in reachable:
                if (pmask & mask) == 0:
                    nmask = pmask | mask
                    if nmask not in new_takes:
                        new_takes[nmask] = pmask
                        new_reachable.add(nmask)
            reachable = new_reachable
            takes[k] = new_takes
            if full_mask in reachable:
                found_k = k
                break
        
        if found_k == -1:
            return [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17]]
        
        result_triples = []
        cur_mask = full_mask
        for k in range(found_k, 0, -1):
            if cur_mask in takes[k]:
                pmask = takes[k][cur_mask]
                idx, mask, tri_list = unique_triples[k-1]
                result_triples.append(tri_list)
                cur_mask = pmask
        
        best_result = result_triples
        best_score = unique_triples[found_k-1][0]
    else:
        # We have an upper bound, now binary search for the minimum
        lo = 1
        hi = best_score
        
        while lo < hi and time.time() < deadline - 10:
            mid = (lo + hi) // 2
            res = greedy_match(mid, deadline - 5)
            if res is not None:
                best_result = res
                best_score = mid
                hi = mid
            else:
                lo = mid + 1
        
        # Final verification and polish
        if time.time() < deadline - 5:
            # Try to verify and potentially improve with exact backtracking
            res = can_match(hi, deadline - 2)
            if res is not None:
                best_result = res
                best_score = hi

    # Ensure correct types
    result = []
    for tri in best_result:
        result.append([int(v) for v in tri])
    
    return result
