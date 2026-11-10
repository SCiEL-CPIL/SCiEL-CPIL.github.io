"""Program for dense odd-order graphs avoiding equal-degree path endpoints."""

# EVOLVE-BLOCK-START
import random

P = {'n': 15}

def construct():
    n = P['n']
    
    # Target: Maximize edges with distinct degrees as much as possible.
    # For n=15, max distinct degrees is 14. One degree must be repeated.
    # Base degrees 1..14. Sum 105.
    # Repeat an odd degree k to make sum even.
    # Candidates for k: 13, 11, 9, 7, 5, 3, 1.
    # Higher k -> Higher sum -> More edges.
    
    candidates = [13, 11, 9, 7, 5, 3, 1]
    
    def havel_hakimi_construct(deg_seq):
        """
        Constructs a graph given a degree sequence.
        Returns adjacency matrix or None if not graphic.
        Uses a greedy deterministic approach for stability, 
        but randomization might help if we want to explore multiple structures.
        For now, deterministic HH.
        """
        n = len(deg_seq)
        # Pair (degree, original_index)
        vertices = sorted([(d, i) for i, d in enumerate(deg_seq)], reverse=True)
        adj = [[False] * n for _ in range(n)]
        
        for deg, u in vertices:
            if deg == 0:
                continue
            # Find 'deg' other vertices with highest degrees
            # We need to pick 'deg' vertices from the remaining pool
            # Standard HH: pick the next 'deg' vertices in the sorted list
            # But we must be careful with indices
            
            # Create list of available vertices (excluding u)
            # In standard HH, we just take the next ones.
            # However, we need to update the degrees.
            pass
            
        # Re-implementing HH properly for construction
        def hh_construct(seq):
            n = len(seq)
            adj = [[False] * n for _ in range(n)]
            # List of (current_degree, index)
            curr = sorted([(d, i) for i, d in enumerate(seq)], reverse=True)
            
            for i in range(n):
                d, u = curr[i]
                if d < 0:
                    return None
                if d == 0:
                    continue
                # We need to connect u to d other vertices
                # These should be the vertices with the highest current degrees
                # among those with index > i in the sorted list?
                # Actually, HH says: remove d, subtract 1 from the next d elements.
                # So we connect u to curr[i+1] ... curr[i+d]
                
                if i + d >= n:
                    return None
                
                targets = []
                for j in range(i + 1, i + d + 1):
                    td, tv = curr[j]
                    if td <= 0:
                        return None
                    targets.append(tv)
                    # Update degree of target
                    # We need to update the 'curr' list
                    # This simple in-place update is tricky because the list is sorted.
                    # We will use a different approach: maintain a list of degrees and indices,
                    # and at each step, pick the top 'd' available.
                    pass
                
            # Let's use a simpler, robust HH construction
            def construct_from_seq(seq):
                n = len(seq)
                adj = [[False] * n for _ in range(n)]
                # (degree, index)
                avail = [(d, i) for i, d in enumerate(seq)]
                avail.sort(reverse=True)
                
                for k in range(n):
                    if not avail:
                        break
                    d, u = avail.pop(0)
                    if d < 0:
                        return None
                    if d == 0:
                        continue
                    
                    # We need to pick d vertices from avail
                    # To ensure validity, we should pick the ones with highest degrees
                    # But we must also ensure we don't pick more than available
                    if len(avail) < d:
                        return None
                    
                    # Pick the first d from the sorted avail
                    # This is the standard greedy HH step
                    chosen = []
                    for _ in range(d):
                        td, tv = avail.pop(0)
                        if td <= 0:
                            return None
                        chosen.append((td - 1, tv))
                        adj[u][tv] = adj[tv][u] = True
                    
                    # Re-insert the chosen vertices with updated degrees
                    # And re-sort
                    avail.extend(chosen)
                    avail.sort(reverse=True)
                
                # Check if all remaining are 0
                for d, i in avail:
                    if d != 0:
                        return None
                        # Note: In standard HH, if we reach the end, 
                        # the remaining must be 0.
                        # But my loop pops one by one.
                        # If I pop a 0, I skip.
                        # If I pop a negative, I fail.
                        # If I finish the loop, I need to ensure no positive left.
                        # Actually, the loop runs n times.
                        # If valid, all will be processed.
                
                return adj

            return construct_from_seq(seq)

        return hh_construct(deg_seq)

    def check_valid(adj, n):
        """Check if any simple 3-edge path s-u-v-t has deg(s)==deg(t)."""
        deg = [sum(adj[i]) for i in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    best_edges = []
    best_count = -1

    for k in candidates:
        # Construct degree sequence
        base = list(range(1, n)) # 1 to 14
        seq = base + [k]
        # Shuffle? No, HH is deterministic on sorted seq.
        # But the resulting graph structure depends on the order of ties?
        # HH is deterministic if we break ties consistently.
        # We can try a few random shuffles of the indices if we want to explore
        # different isomorphic copies, but the degree sequence is the same.
        # The constraint is on the structure.
        # Different HH runs (if randomized) might yield different structures.
        
        # Let's try a few random permutations of the sequence indices
        # to see if we can find a valid structure.
        found_valid = False
        for attempt in range(10):
            # Shuffle the assignment of degrees to vertices
            # This changes which vertex gets which degree, potentially changing
            # the relative positions of the two high-degree vertices.
            random.shuffle(seq)
            adj = havel_hakimi_construct(seq)
            if adj is not None:
                if check_valid(adj, n):
                    # Count edges
                    count = sum(sum(row) for row in adj) // 2
                    if count > best_count:
                        best_count = count
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        best_edges = edges
                    found_valid = True
                    break # Found one valid for this k, move to next k (lower k has fewer edges)
        
        if found_valid:
            break # Since k is decreasing, the first valid one is the best

    if best_count == -1:
        # Fallback to greedy if all fail (should not happen)
        return construct_fallback()

    # Convert to proper Python ints
    edges = [[int(u), int(v)] for u, v in best_edges]
    return edges

def construct_fallback():
    n = 15
    adj = [[False] * n for _ in range(n)]
    for v in range(1, n):
        adj[0][v] = adj[v][0] = True
    
    def get_degrees():
        return [sum(adj[i]) for i in range(n)]
    
    def check_valid(adj, n):
        deg = get_degrees()
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    added = True
    while added:
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    continue
                adj[i][j] = adj[j][i] = True
                if check_valid(adj, n):
                    added = True
                    break
                else:
                    adj[i][j] = adj[j][i] = False
            if added:
                break

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END
