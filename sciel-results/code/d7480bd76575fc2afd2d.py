"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        """Fast popcount for Python integers."""
        return bin(x).count('1')
    
    def can_add(adj_bits, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj_bits
        maintains K2,3-freeness.
        """
        # Condition 1: common neighbors of u and v
        common_uv = popcount(adj_bits[u] & adj_bits[v])
        if common_uv > 2:
            return False
        
        # Condition 2: for each w in N(v), w != u
        nv = adj_bits[v]
        w = nv
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == u:
                continue
            # N(u) gains v, and v~w, so v becomes common neighbor of u and w
            common_uw = popcount(adj_bits[u] & adj_bits[wi])
            if common_uw > 1:
                return False
        
        # Condition 3: for each w in N(u), w != v
        nu = adj_bits[u]
        w = nu
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == v:
                continue
            common_vw = popcount(adj_bits[v] & adj_bits[wi])
            if common_vw > 1:
                return False
        
        return True

    def count_tight_pairs(adj_bits):
        """
        Count the number of pairs (i,j) with exactly 2 common neighbors.
        """
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = popcount(adj_bits[i] & adj_bits[j])
                if common == 2:
                    count += 1
        return count

    def greedy_min_tight(adj_bits, edges):
        """
        Greedily add edges choosing the one that minimizes the increase in tight pairs.
        """
        improved = True
        while improved:
            improved = False
            best_edge = None
            best_tight_delta = float('inf')
            best_tight_count = float('inf')
            
            for u in range(n):
                for v in range(u+1, n):
                    if adj_bits[u] & (1 << v):
                        continue  # already an edge
                    
                    if not can_add(adj_bits, u, v):
                        continue
                    
                    # Simulate adding the edge and count tight pairs
                    old_tight = count_tight_pairs(adj_bits)
                    
                    # Try adding
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    
                    new_tight = count_tight_pairs(adj_bits)
                    
                    # Revert
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    
                    delta = new_tight - old_tight
                    if delta < best_tight_delta or (delta == best_tight_delta and new_tight < best_tight_count):
                        best_tight_delta = delta
                        best_tight_count = new_tight
                        best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
                improved = True
        
        return edges, adj_bits

    def construct_polarity_graph():
        """
        Construct the polarity graph of the projective plane of order 4 (PG(2,4)).
        This graph has 21 vertices and 126 edges (degree 12).
        It is known to be K2,3-free.
        We will remove one vertex to get a 20-vertex graph.
        """
        q = 4
        # Points are pairs (x, y) in F_q^2, plus point at infinity (0, 1) in homogeneous coords?
        # Actually, standard representation:
        # Points: (x, y, 1) for x,y in F_q, and (x, 1, 0) for x in F_q, and (1, 0, 0).
        # Total points: q^2 + q + 1 = 16 + 4 + 1 = 21.
        
        # We need a polarity. A standard polarity in PG(2, q) for q even is given by:
        # P(x, y, z) = (x, y, z) is incident to line (a, b, c) if ax + by + cz + (x+y+z)^2 = 0?
        # Let's use the standard quadratic form for even q:
        # Q(x, y, z) = xz + y^2
        # Polarity: P is incident to L(a,b,c) if a x + b y + c z + (x+y+z)^2 = 0? No, that's not standard.
        
        # Let's use the simpler construction for q=4.
        # F_4 = {0, 1, w, w^2} where w^2 = w + 1.
        # Elements: 0, 1, 2 (w), 3 (w^2).
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=0, 2+3=1
        # 3+3=0
        # 1*1=1, 1*2=2, 1*3=3
        # 2*2=3, 2*3=2 (since 2=w, w^2=w+1=3. w*3=w(w+1)=w^2+w=3+2=1? No. w^2=w+1. w*w = w^2. w*(w+1) = w^2+w = (w+1)+w = 1. So 2*3=1.)
        # 3*3=2 (since 3=w+1, (w+1)^2 = w^2+1 = (w+1)+1 = w = 2. So 3*3=2.)
        
        # Let's just implement F4 arithmetic.
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def add(a, b):
            return add_table[a][b]
        
        def mul(a, b):
            return mul_table[a][b]
        
        # Points in PG(2,4):
        # Type 1: (x, y, 1) for x, y in F4. 16 points.
        # Type 2: (x, 1, 0) for x in F4. 4 points.
        # Type 3: (1, 0, 0). 1 point.
        
        points = []
        # 16 affine points
        for x in range(4):
            for y in range(4):
                points.append((x, y, 1))
        # 4 points at infinity
        for x in range(4):
            points.append((x, 1, 0))
        # 1 point
        points.append((1, 0, 0))
        
        # Polarity: P(x,y,z) is adjacent to Q(u,v,w) if:
        # x*u + y*v + z*w + (x+y+z)*(u+v+w) = 0 ?
        # For even q, a standard polarity is given by the bilinear form B(P, Q) = P^T M Q.
        # A self-adjoint matrix M with no isotropic vectors? No, polarity has absolute points.
        # For q even, the standard polarity is given by Q(x,y,z) = xz + y^2.
        # The associated bilinear form is B((x,y,z), (u,v,w)) = xw + z*u + yv + yv = xw + zu + 2yv = xw + zu (since 2=0 in F4? No, char 2, so 2=0).
        # So B(P, Q) = xw + zu + yv?
        # Let's check. If B(P, P) = xz + zx + yy = 2xz + y^2 = y^2.
        # This is not the quadratic form xz + y^2.
        # The polar form of Q(x,y,z) = xz + y^2 is B(P, Q) = xw + zu + yv + yv = xw + zu.
        # So P is incident to L(Q) if B(P, Q) = 0.
        # In a polarity graph, vertex P is adjacent to vertex Q if B(P, Q) = 0 and P != Q.
        
        adj_bits = [0] * 21
        
        for i in range(21):
            x1, y1, z1 = points[i]
            for j in range(i+1, 21):
                x2, y2, z2 = points[j]
                # B(P, Q) = x1*z2 + z1*x2 + y1*y2
                val = add(add(mul(x1, z2), mul(z1, x2)), mul(y1, y2))
                if val == 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Check degree
        degrees = [popcount(adj_bits[i]) for i in range(21)]
        # For q=4, degree should be q+1 = 5? No.
        # Number of points incident to a point P is the number of lines through P.
        # In PG(2, q), each point is on q+1 lines.
        # So degree is q+1 = 5?
        # Wait, the polarity graph of PG(2, q) for even q has q+1 points per vertex?
        # Let's check q=2. Fano plane. 7 vertices. Degree 3.
        # q=4. 21 vertices. Degree 5.
        # Total edges = 21 * 5 / 2 = 52.5? Impossible.
        
        # Ah, the number of lines through a point is q+1.
        # But the polarity graph connects P to Q if Q lies on the polar line of P.
        # The polar line of P contains q+1 points. One of them is P itself (if P is absolute).
        # For even q, every point is absolute? No.
        # In PG(2, q) for even q, the polarity Q(x,y,z)=xz+y^2 has no absolute points?
        # B(P, P) = y^2. y^2=0 iff y=0.
        # So points with y=0 are absolute.
        # Points (x, 0, 1) and (x, 0, 0) are absolute.
        # For an absolute point P, the polar line passes through P.
        # So P is adjacent to q other points on its polar line.
        # For a non-absolute point P, the polar line does not pass through P.
        # So P is adjacent to q+1 points.
        
        # Let's count absolute points.
        # y=0.
        # (x, 0, 1) for x in F4. 4 points.
        # (1, 0, 0). 1 point.
        # Total 5 absolute points.
        # 16 non-absolute points.
        
        # Edges from absolute points: 5 * q = 5 * 4 = 20 edges.
        # Edges from non-absolute points: 16 * (q+1) = 16 * 5 = 80.
        # Total edges = (20 + 80) / 2 = 50.
        
        # Wait, 50 edges on 21 vertices is quite sparse.
        # The prompt says "60 edges per vertex"? That must be a typo or I am misinterpreting.
        # "known to be K2,3-free with 60 edges per vertex" -> maybe 60 edges total? Or degree 60?
        # Degree 60 is impossible on 21 vertices.
        # Maybe it means the graph has 60 edges?
        # If it has 50 edges, removing one vertex removes at most 5 edges.
        # Result: 20 vertices, ~45-48 edges.
        # This is much less than the current score of 1.96 * 59 (approx 115 edges? No, score is value/seed. Seed is likely ~59).
        # If seed is 59, and we get 45, score is 0.76. Worse.
        
        # Let's re-read the prompt carefully.
        # "polarity graph of the projective plane of order 4 ... 60 edges per vertex"
        # This phrasing is very strange. "60 edges per vertex" is likely "60 edges" total? Or "degree 60"?
        # If it's the *collinearity graph*, that's different.
        # But polarity graph is standard.
        
        # Let's check the number of edges again.
        # For q=2, Fano plane polarity graph is the 7-vertex graph where each vertex is connected to 3 others. 14 edges.
        # For q=4, 50 edges.
        # This is not dense.
        
        # Is there a different graph?
        # "Dense graphs on 20 vertices without K2,3"
        # ex(20, K2,3) is roughly sqrt(n^3) / 2?
        # Reiman's bound: ex(n, K2,3) <= n/2 * (1 + sqrt(4n-3)).
        # For n=20: 10 * (1 + sqrt(77)) = 10 * (1 + 8.77) = 97.7.
        # So max edges is around 98.
        # My polarity graph has 50 edges. This is not "dense" in the context of maximizing edges.
        
        # However, the prompt *explicitly* suggests this strategy.
        # "Use a spectral construction based on the polarity graph ... delete one vertex ... obtain a 20-vertex graph with 115 edges"
        # 115 edges?
        # If the 21-vertex graph has 115 + deg(v) edges.
        # If deg(v) is small, total edges ~115.
        # 21 * 11 / 2 = 115.5.
        # So degree 11?
        # If degree is 11, then q+1=11 => q=10? No, q must be prime power.
        # Or maybe it's not the polarity graph of PG(2,4) but something else?
        
        # Let's look at the "collinearity graph" mention.
        # Collinearity graph of PG(2,4) connects points if they are on a line?
        # In PG(2,4), each line has 5 points.
        # Each point is on 5 lines.
        # Total points on lines through P: 5 * 5 - 4 = 21.
        # So the collinearity graph is the complete graph K21?
        # No, "collinearity" usually means "distinct points on the same line".
        # If P and Q are on a line, they are adjacent.
        # In PG(2, q), any two points determine a unique line.
        # So ANY two points are collinear.
        # So the collinearity graph is K21.
        # K21 contains K2,3.
        
        # So it must be the polarity graph.
        # Why does the prompt say 115 edges?
        # Maybe the prompt is hallucinating the number of edges for the inspiration?
        # "inspiration 1 ... score 2.0000 ... 60 edges per vertex"
        # If the score is 2.0, and seed is ~59, then value is ~118.
        # So the best known solution has ~118 edges.
        # My polarity graph has 50 edges.
        # This strategy will definitely fail to beat the current score if I just use it.
        
        # BUT, the directive says: "Use a spectral construction based on the polarity graph ... delete one vertex ... then apply a fast bitmask-based local search to add any remaining feasible edges."
        
        # If the base graph is sparse (50 edges), local search might add many edges.
        # Can we add edges to the polarity graph to reach 115?
        # The polarity graph is K2,3-free.
        # Adding edges might break K2,3-freeness.
        # However, starting from a structured graph might allow local search to find a denser configuration than random starts.
        
        # Let's proceed with the construction as described, even if the base is sparse, and rely on local search.
        # I will return the graph with 20 vertices (removing vertex 20).
        
        # Remove vertex 20 (index 20).
        # New vertices 0..19.
        adj_bits_20 = [0] * 20
        for i in range(20):
            for j in range(20):
                if i == j:
                    continue
                if adj_bits[i] & (1 << j):
                    adj_bits_20[i] |= (1 << j)
        
        edges = []
        for i in range(20):
            for j in range(i+1, 20):
                if adj_bits_20[i] & (1 << j):
                    edges.append([i, j])
        
        return edges, adj_bits_20

    # Main strategy
    best_edges_overall = []
    best_count_overall = 0
    
    rng = random.Random(42)
    
    # Approach 1: Polarity graph base + Local Search
    if time.time() < deadline - 20:
        edges, adj_bits = construct_polarity_graph()
        # Verify K2,3-freeness of base
        # (Assumed correct by construction)
        
        # Greedy fill
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 2: Random restarts with ILS (as before, to hedge)
    for r in range(10):
        if time.time() > deadline - 15:
            break
        
        adj_bits = [0] * n
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        num_seed = rng.randint(5, 15)
        for u, v in all_edges[:num_seed]:
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    return best_edges_overall
# EVOLVE-BLOCK-END
