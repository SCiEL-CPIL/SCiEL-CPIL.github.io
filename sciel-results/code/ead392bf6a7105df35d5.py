import time
import itertools
import random

def _has_c5_fast(n, adj_mat):
    """Check if graph has a C5. adj_mat is a list of bitmasks (ints).
    Returns True if C5 exists."""
    # For each 5-tuple (a,b,c,d,e) with a<b<c<d<e, check if any ordering forms a cycle.
    # To be efficient, we check all 12 distinct 5-cycles on 5 labeled vertices.
    # But actually, we can be smarter: fix a as the smallest, then check permutations of the other 4.
    # For n=35, C(35,5) = 324632, and 12 checks each = ~3.9M operations.
    # With bitmasks, each check is a few bit operations.
    
    # The 12 cycles on vertices (a,b,c,d,e) can be enumerated by fixing a and
    # permuting (b,c,d,e), but each cycle is counted twice (reversal), so 24/2=12.
    # Actually, let's just check all 24 permutations and break early.
    
    for a, b, c, d, e in itertools.combinations(range(n), 5):
        # Check all permutations of (b,c,d,e) as the middle of the path a-?-?-?-?-a
        # We need: a-p1, p1-p2, p2-p3, p3-p4, p4-a all edges
        for p1, p2, p3, p4 in itertools.permutations((b, c, d, e)):
            if (adj_mat[a] >> p1) & 1:
                if (adj_mat[p1] >> p2) & 1:
                    if (adj_mat[p2] >> p3) & 1:
                        if (adj_mat[p3] >> p4) & 1:
                            if (adj_mat[p4] >> a) & 1:
                                return True
    return False

def _has_independent_set_k(n, adj_mat, k=9):
    """Check if graph has independent set of size k. adj_mat is list of bitmasks.
    Returns True if such a set exists."""
    # Use backtracking with bitmask representation
    # candidates is a bitmask of vertices that can still be added
    # We want to find k vertices no two adjacent
    
    # Precompute: for each vertex, its non-neighbors (including itself) as bitmask
    # Actually, for independent set, when we pick vertex v, we remove v and all its neighbors
    # from candidates. So the new candidates = candidates & ~(adj_mat[v] | (1 << v))
    
    # Use a recursive search with pruning
    found = [False]
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [bin(adj_mat[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: degrees[v])
    
    def search(chosen_count, candidates):
        if found[0]:
            return
        if chosen_count >= k:
            found[0] = True
            return
        # Pruning: if even adding all candidates can't reach k
        if chosen_count + bin(candidates).count('1') < k:
            return
        if candidates == 0:
            return
        
        # Pick a vertex from candidates
        # Choose the one with minimum degree within candidates
        # For speed, just pick the lowest set bit or use a heuristic
        # Let's pick the vertex with minimum degree in the candidate subgraph
        best_v = -1
        best_deg = float('inf')
        temp = candidates
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            # Count neighbors of v within candidates
            deg = bin(adj_mat[v] & candidates).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        bit = 1 << v
        
        # Branch 1: include v
        new_candidates = candidates & ~(adj_mat[v] | bit)
        search(chosen_count + 1, new_candidates)
        if found[0]:
            return
        
        # Branch 2: exclude v
        search(chosen_count, candidates & ~bit)
    
    search(0, (1 << n) - 1)
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Base: 8 disjoint K4s, n=32
    def make_8k4_edges():
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return edges
    
    best_n = 32
    best_edges = make_8k4_edges()
    
    def verify(n, edges):
        """Verify C5-free and alpha <= 8. Returns (c5_free, alpha_ok)."""
        adj_mat = [0] * n
        for u, v in edges:
            adj_mat[u] |= (1 << v)
            adj_mat[v] |= (1 << u)
        
        # Check C5-free
        if _has_c5_fast(n, adj_mat):
            return False, True  # Has C5, invalid
        
        # Check alpha <= 8 (i.e., no independent set of size 9)
        if _has_independent_set_k(n, adj_mat, 9):
            return True, False  # C5-free but has alpha >= 9
        
        return True, True  # Valid
    
    # Try n=33, 34, 35 with structured constructions
    # Strategy: 8 K4s + extra vertices, each extra vertex connected to all 4 in one K4
    # and to one vertex in each other K4
    
    # For n=33: vertex 32 connected to all of K4_0 and one in each K4_1..K4_7
    def try_n33():
        edges = make_8k4_edges()
        # Vertex 32: connect to all 4 in K4_0 (0,1,2,3)
        for v in range(4):
            edges.append([32, v])
        # Connect to one in each K4_1..K4_7
        for b in range(1, 8):
            edges.append([32, 4*b])
        return edges
    
    # For n=34: add vertex 33 connected to all of K4_1 and one in each other K4
    def try_n34():
        edges = try_n33()
        # Vertex 33: connect to all 4 in K4_1 (4,5,6,7)
        for v in range(4, 8):
            edges.append([33, v])
        # Connect to one in each other K4
        edges.append([33, 0])  # K4_0
        for b in range(2, 8):
            edges.append([33, 4*b])
        return edges
    
    # For n=35: add vertex 34 connected to all of K4_2 and one in each other K4
    def try_n35():
        edges = try_n34()
        # Vertex 34: connect to all 4 in K4_2 (8,9,10,11)
        for v in range(8, 12):
            edges.append([34, v])
        # Connect to one in each other K4
        edges.append([34, 0])   # K4_0
        edges.append([34, 4])   # K4_1
        for b in range(3, 8):
            edges.append([34, 4*b])
        return edges
    
    # Try from highest n down
    for target_n, builder in [(35, try_n35), (34, try_n34), (33, try_n33)]:
        if time.time() >= deadline - 5:
            break
        if target_n <= best_n:
            continue
        
        edges = builder()
        c5_free, alpha_ok = verify(target_n, edges)
        if c5_free and alpha_ok:
            best_n = target_n
            best_edges = edges
            break  # Found the highest possible, done
    
    # If structured constructions didn't work, try random search
    if best_n < 35 and time.time() < deadline - 10:
        for target_n in [35, 34, 33]:
            if target_n <= best_n:
                continue
            if time.time() >= deadline - 5:
                break
            
            n_extra = target_n - 32
            # Try multiple random constructions
            for trial in range(50):
                if time.time() >= deadline - 5:
                    break
                
                random.seed(trial * 7919 + target_n)
                edges = make_8k4_edges()
                
                for k in range(n_extra):
                    v = 32 + k
                    # Connect to all 4 in one random K4
                    chosen_k4 = random.randint(0, 7)
                    base = 4 * chosen_k4
                    for i in range(4):
                        edges.append([v, base + i])
                    # Connect to one random vertex in each other K4
                    for b in range(8):
                        if b != chosen_k4:
                            edges.append([v, 4*b + random.randint(0, 3)])
                    # Connect to other extra vertices
                    for w in range(32, 32+k):
                        if random.random() < 0.5:
                            edges.append([v, w])
                
                c5_free, alpha_ok = verify(target_n, edges)
                if c5_free and alpha_ok:
                    best_n = target_n
                    best_edges = edges
                    break
    
    return {'n': best_n, 'edges': best_edges}
