"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 18
    
    # Construct Clebsch graph on 16 vertices
    # Vertices are binary strings of length 4 with even parity
    # Or simply use the known adjacency list
    # Clebsch graph is the folded 5-cube? No.
    # It is the graph where vertices are 4-bit strings with even weight.
    # Two vertices are adjacent if their Hamming distance is 2?
    # Let's check:
    # Weights 0: 0000 (1)
    # Weights 2: 0011, 0101, 0110, 1001, 1010, 1100 (6)
    # Weights 4: 1111 (1)
    # Total 8? No, 1+6+1=8.
    # Clebsch graph has 16 vertices.
    # It is the **halved 5-cube**? No.
    # It is the **complement** of the **line graph of K_8**? No.
    # Let's use a standard generation.
    # Clebsch graph is strongly regular (16, 5, 0, 2).
    # We can generate it using the 4-dimensional hypercube Q4.
    # Vertices of Q4 are 0..15.
    # Adjacency in Clebsch: two vertices are adjacent if their Hamming distance is 1 or 3?
    # No, that's the complement of the Clebsch graph?
    # Let's try: Adjacent if dist == 1. That's Q4. Q4 is bipartite, triangle-free.
    # Q4 has 16 vertices.
    # Is Q4 the Clebsch graph?
    # Q4 is (16, 4, 0, 2)? No, degree 4.
    # Clebsch is degree 5.
    # So Clebsch is not Q4.
    # Clebsch is obtained from Q4 by adding edges between vertices at distance 3?
    # If we add edges between dist 3, degree becomes 4+1=5?
    # In Q4, each vertex has 1 vertex at dist 3? No, 4 vertices at dist 3?
    # Binomial(4,3)=4. So degree would be 4+4=8.
    
    # Let's use a known list of edges for Clebsch graph.
    # Or generate it as the **folded 5-cube**?
    # The folded 5-cube has 16 vertices.
    # Vertices are binary strings of length 5 with even parity? No.
    # Folded 5-cube: Identify antipodal vertices of Q5.
    # Q5 has 32 vertices. 32/2 = 16.
    # Yes, the Clebsch graph is the folded 5-cube.
    
    # Generate Q5 vertices (0..31)
    # Fold: u ~ v if v = u XOR 31 (11111)
    # We can represent vertices by 0..15.
    # u is adjacent to v in Clebsch if u XOR v is a weight 1 or 2 vector in Q5?
    # In Q5, edges connect weight 1 differences.
    # In folded Q5, edges connect weight 1 differences OR weight 4 differences (which are weight 1 in complement)?
    # Actually, in folded Q5, two vertices are adjacent if their Hamming distance is 1 or 4.
    # Let's verify degree.
    # For a vertex u, there are 5 vertices at dist 1.
    # There are 5 vertices at dist 4.
    # Total 10? No, Clebsch is degree 5.
    # Wait, the folded 5-cube has degree 5.
    # The adjacency is: u ~ v if dist(u,v) = 1 or dist(u,v) = 4?
    # If u=00000.
    # Dist 1: 10000, 01000, 00100, 00010, 00001. (5 vertices)
    # Dist 4: 11110, 11101, 11011, 10111, 01111. (5 vertices)
    # But in the folded graph, 11110 is identified with 00001.
    # So the set of neighbors is {10000, 01000, 00100, 00010, 00001}.
    # Wait, 00001 is identified with 11110.
    # So the neighbors are the 5 weight-1 vectors.
    # So the Clebsch graph is isomorphic to the graph on 16 vertices where u~v if dist(u,v) in {1, 4}?
    # No, if we identify antipodes, the neighbors of 0 are the 5 weight-1 vectors.
    # So it is just the graph induced by the 5 weight-1 vectors?
    # No, the vertices are the 16 equivalence classes.
    # Let's label vertices 0..15.
    # v is adjacent to u if (u XOR v) has weight 1 or 2?
    # Let's check the parameters of the graph defined by dist 1 or 2 in Q4?
    # No, let's just use a hardcoded adjacency list for the Clebsch graph to be safe.
    
    # Clebsch graph edges (0-15)
    # Source: OEIS or standard graph library
    clebsch_edges = [
        (0, 1), (0, 2), (0, 4), (0, 8), (0, 16) # Wait, 16 vertices, 0-15
    ]
    
    # Let's generate it properly.
    # Vertices are 0..15.
    # u ~ v if popcount(u ^ v) in [1, 2]?
    # If dist 1: 4 neighbors.
    # If dist 2: 6 neighbors.
    # Total 10. Too many.
    
    # Correct construction:
    # Vertices are binary strings of length 4.
    # u ~ v if dist(u,v) = 1 or dist(u,v) = 3?
    # Dist 1: 4 neighbors.
    # Dist 3: 4 neighbors.
    # Total 8.
    
    # It seems I am struggling to recall the exact Clebsch definition.
    # Let's use the **complement of the line graph of K_8**?
    # L(K_8) has 28 vertices.
    
    # Alternative: Use the **local search** from the seed, but with a better starting point.
    # The seed starts with K_6,6,6 (score 12).
    # If we can find a graph with score 8-10, that's good.
    
    # Let's try to construct a graph based on the **Shrikhande graph** or similar?
    
    # Given the time constraint, I will stick to the local search but improve the heuristic.
    # The main issue with the seed is that it might get stuck.
    # I will add a restart mechanism and a better K4 check.
    
    def is_k4_free(edge_set, n=18):
        # Fast check: for each edge (u,v), check if any common neighbor w has an edge to x?
        # Simpler: iterate all K4s. 18 choose 4 is 3060.
        # 3060 * 6 checks is fast.
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if all((min(p[0],p[1]), max(p[0],p[1])) in edge_set for p in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]):
                return False
        return True

    def max_triangle_free_size(edge_set, n=18, timeout=2.0):
        best = 0
        start = time.time()
        
        def bt(remaining, current_size, current_set):
            nonlocal best
            if time.time() - start > timeout:
                return
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            can_add = True
            for i, u in enumerate(current_set):
                if (min(u,v), max(u,v)) in edge_set:
                    for w in current_set[i+1:]:
                        if (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                bt(rest, current_size + 1, current_set + [v])
            
            bt(rest, current_size, current_set)
        
        bt(list(range(n)), 0, [])
        return best

    def score(edge_set):
        return max_triangle_free_size(edge_set)

    random.seed(42)
    
    # Start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a,b), max(a,b)))
    
    current_edges = set(edges)
    current_score = score(current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    all_possible_edges = set((i,j) for i in range(n) for j in range(i+1, n))
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        # Random restarts if stuck
        if iteration % 20 == 0 and current_score > 8:
            # Try a different random graph
            e = set()
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        e.add((i,j))
            if is_k4_free(e):
                s = score(e)
                if s < current_score:
                    current_edges = e
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    continue
        
        # Local search
        candidates = list(all_possible_edges - current_edges) + list(current_edges)
        random.shuffle(candidates)
        
        improved = False
        for e in candidates[:30]:
            if time.time() > deadline:
                break
            if e in current_edges:
                new_edges = current_edges - {e}
            else:
                new_edges = current_edges | {e}
                if not is_k4_free(new_edges):
                    continue
            
            s = score(new_edges)
            if s < current_score:
                current_edges = new_edges
                current_score = s
                improved = True
                if s < best_score:
                    best_score = s
                    best_edges = set(current_edges)
                break
        
        if not improved:
            # Perturb
            for _ in range(5):
                e = random.choice(list(all_possible_edges))
                if e in current_edges:
                    current_edges.remove(e)
                else:
                    if is_k4_free(current_edges | {e}):
                        current_edges.add(e)
                current_score = score(current_edges)
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
    
    return [[u, v] for u, v in sorted(best_edges)]
# EVOLVE-BLOCK-END
