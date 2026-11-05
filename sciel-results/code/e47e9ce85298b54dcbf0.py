import time
import random
import itertools

# Precompute Q3 structure
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with sorted unassigned vertices
# This helps the DFS prune earlier by trying "constrained" vertices first
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        # Sort unassigned vertices by how many neighbors they have in the Q3 
        # relative to the assigned vertices? 
        # The inspiration used a heuristic: _cc(x, a, b) counts neighbors in Q3.
        # We want to assign vertices that have many constraints early?
        # Actually, the inspiration sorted by _cc descending.
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    n = 16
    full_mask = (1 << n) - 1
    
    best_edges = None
    best_count = 0
    
    # We will try multiple restarts with different random seeds for shuffling
    # to escape local optima of the greedy algorithm.
    
    # Initial seed: K_{3,13}
    # Part A: 0, 1, 2. Part B: 3..15
    # Edges: all (a, b) where a in A, b in B.
    
    def get_initial_adj():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def has_q3_after_add(u, v, adj):
        """
        Check if adding edge (u, v) to graph defined by adj masks creates a Q3 subgraph.
        Returns True if Q3 is created (so we should NOT add the edge).
        """
        # Temporarily add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        found = False
        
        # Try all orientations of Q3 edges
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            # Try both mappings of (u, v) to (qa, qb) and (qb, qa)
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                # DFS to assign remaining 6 vertices
                def bt(idx, used_mask):
                    if time.time() > deadline - 5.0:
                        return False
                    if idx == 6:
                        # All assigned. Check if all Q3 edges exist in G
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    # Candidates: vertices not used, and adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    found = True
                    break
            
            if found:
                break

        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        return found

    def run_greedy(seed_val):
        rng = random.Random(seed_val)
        adj = get_initial_adj()
        
        # Generate list of missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
        
        rng.shuffle(missing)
        
        # Greedy addition
        for u, v in missing:
            if time.time() > deadline - 10.0:
                break
            if not has_q3_after_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local Search: Try to add any remaining missing edges
        # This helps if the random order skipped some valid edges early on
        # but they became valid later? No, adding edges only makes it harder to add more.
        # So if an edge was invalid before, it's still invalid now.
        # However, if we shuffle, we might add edges in a different order.
        # The first pass is the main driver.
        
        # Count edges
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        count //= 2
        
        if count > best_count:
            # Store a copy
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return count, edges
        return count, None

    # Run multiple iterations
    # Each iteration takes some time. We run as many as possible.
    current_seed = 0
    while time.time() < deadline - 20.0:
        cnt, edges = run_greedy(current_seed)
        if edges is not None:
            best_count = cnt
            best_edges = edges
        
        current_seed += 1
        
        # If we hit a very high score, we might want to break early?
        # Max possible edges is 120. Q3-free max is likely around 50-60.
        # If best_count > 55, we are doing well.
        if best_count > 60:
            break

    if best_edges is None:
        # Fallback to initial K_{3,13}
        adj = get_initial_adj()
        best_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    best_edges.append([i, j])
        best_count = 39

    return best_edges
