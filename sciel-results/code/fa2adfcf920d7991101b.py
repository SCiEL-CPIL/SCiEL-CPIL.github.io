import numpy as np
import time
import random
import math

def count_c4_incremental(adj, edge_to_add=None, edge_to_remove=None):
    """
    Efficiently count C4s. 
    If no edges change, just compute from scratch (fast enough for 31x31).
    If edges change, we can use the property:
    C4 count = 1/2 * sum_{u<v} (common_neighbors(u,v) choose 2)
    Actually, the standard formula is:
    C4 = 1/2 * sum_{u} (deg(u) choose 2) - sum_{triangles} ... no that's for triangles.
    
    Let's stick to the A^2 method but optimize the counting.
    For n=31, A^2 is very fast. 
    However, for simulated annealing, we need to evaluate many swaps.
    
    Let's use the formula:
    C4(G) = 1/2 * sum_{u < v} ( (A^2)_{uv} choose 2 )
    
    When we swap (u1, v1) with (u2, v2):
    The change in C4 is local to the neighbors of u1, v1, u2, v2.
    But implementing incremental delta for C4 is complex.
    Given n=31, a full recount via numpy is ~microseconds.
    Let's check speed: 31x31 matmul is trivial.
    We will use the full recount but optimize the inner loop of the search.
    """
    A2 = adj @ adj
    mask = np.triu(np.ones((31, 31), dtype=bool), k=1)
    vals = A2[mask]
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit
    
    # 1. Construct the base graph: Polarity graph of PG(2,5)
    # The polarity graph of PG(2, q) has q^2+q+1 vertices and q(q+1) edges (if q is odd, it has q^2+q+1 vertices and q(q+1) edges? No.)
    # For q=5 (odd), the polarity graph has 31 vertices.
    # The number of edges in the polarity graph of PG(2, q) for odd q is q(q+1).
    # 5 * 6 = 30? No.
    # Let's recall: ex(n, C4) for n = q^2+q+1 is q(q+1) for q odd?
    # ex(31, C4) = 90.
    # 90 = 5 * 18? No.
    # The polarity graph of PG(2,5) has 31 vertices.
    # Each point is incident to q+1 = 6 lines.
    # In the polarity graph, a point is adjacent to the points on the lines incident to it, plus itself if it's absolute.
    # For odd q, every point is absolute in a Hermitian polarity? No, for unitary polarity.
    # Actually, the standard C4-free graph with 31 vertices and 90 edges is the polarity graph of the unitary polarity of PG(2,5).
    # Let's construct it.
    
    # Coordinates in GF(5)
    # We can represent points of PG(2,5) as triples (x, y, 1) and (x, 0, 0), (0, 1, 0).
    # Total 25 + 5 + 1 = 31 points.
    
    points = []
    # Affine part: (x, y, 1) for x,y in 0..4
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Line at infinity: (x, 0, 0) for x in 0..4
    for x in range(5):
        points.append((x, 0, 0))
    # Point (0, 1, 0)
    points.append((0, 1, 0))
    
    # Define the unitary polarity in PG(2,5).
    # The unitary polarity is defined by the Hermitian form:
    # H(P, Q) = x1*x2^5 + y1*y2^5 + z1*z2^5 ... no, in GF(q^2).
    # Wait, PG(2,5) is over GF(5). The unitary polarity requires GF(25).
    # The graph with 31 vertices and 90 edges is the **Polarity Graph of the Unitary Polarity of PG(2,5)**.
    # It is a well-known extremal graph.
    # Construction:
    # Vertices are points of PG(2,5).
    # Two points P and Q are adjacent if Q lies on the polar line of P.
    # For the unitary polarity, the polar line of P contains P (since P is on its own polar line? No, P is on polar line of P iff P is absolute).
    # In unitary polarity over GF(q^2), all points are absolute.
    # So the graph is undirected.
    # Degree of each vertex: The polar line of P contains q+1 = 6 points. One is P. So P is adjacent to 5 other points?
    # If degree is 5, total edges = 31*5/2 = 77.5. Not integer.
    # Wait, if q is odd, q^2+q+1 is odd.
    # Let's check the degree.
    # In the polarity graph of a non-singular polarity in PG(2,q):
    # If q is odd, the graph has q^2+q+1 vertices.
    # The number of absolute points is q+1.
    # Absolute points have degree q. Non-absolute have degree q+1? Or vice versa?
    # For unitary polarity (q^2=25, so base field GF(5) is not correct for unitary, unitary is over GF(q^2) with conjugation).
    # The graph with 31 vertices and 90 edges is specifically the **Erdos-Renyi** or **Reiman** graph?
    # Actually, ex(31, C4) = 90 is achieved by the **Polarity Graph of PG(2,5)** associated with a **Unitary Polarity**.
    # Let's build the incidence matrix directly using the algebraic definition.
    
    # We work in GF(25) = GF(5)[alpha] where alpha^2+1=0 (since 5 is odd, -1 is not a square in GF(5)? 0^2=0, 1^2=1, 2^2=4, 3^2=9=4, 4^2=16=1. Yes, -1=4 is not a square).
    # So alpha^2 = -1 = 4.
    # Elements of GF(25) are a + b*alpha.
    # Conjugation: (a + b*alpha)^* = a - b*alpha.
    # Hermitian form: H(P, Q) = x1*x2^* + y1*y2^* + z1*z2^* = 0.
    # Points are in PG(2, 25). But we want 31 points, which is PG(2,5).
    # This implies the graph is constructed on the points of PG(2,5) that are "rational" with respect to the unitary polarity?
    # No, the standard result is that the polarity graph of the unitary polarity of PG(2, q) (where q is a prime power) has q^2+q+1 vertices?
    # No, the unitary polarity is on PG(2, q^2). It has q^2+q+1 points? No, PG(2, q^2) has q^4+q^2+1 points.
    
    # Correction: The graph with 31 vertices and 90 edges is the **Polarity Graph of PG(2,5)** with a **Symplectic/Orthogonal** polarity?
    # For odd q, PG(2,q) has a polarity.
    # If we take the polarity graph of PG(2,5) (points are adjacent if they are incident with the polar line of the other),
    # The number of edges is q(q+1) if q is even?
    # For q=5, ex(31, C4) = 90.
    # 90 = 5 * 18.
    # The degree sequence must be regular or semi-regular.
    # 31 * d = 180 => d = 180/31 ~ 5.8.
    # So it's not regular.
    
    # Let's use a known construction: The **Reiman Graph** or simply generate the **Polarity Graph of PG(2,5)** with a standard polarity (e.g. x^2 + y^2 + z^2 = 0 type, but over GF(5)).
    # Over GF(5), x^2+y^2+z^2=0 has solutions.
    # Let's define the polarity by the matrix M = I (identity).
    # P=(x,y,z) is adjacent to Q=(x',y',z') if x*x' + y*y' + z*z' = 0.
    # This is a symmetric bilinear form.
    # Since the form is symmetric, the graph is undirected.
    # P is adjacent to P if x^2+y^2+z^2=0.
    # If P is not on its polar line (x^2+y^2+z^2 != 0), it is not a loop.
    # If P is on its polar line, it is a loop? No, in simple graphs, we ignore loops.
    # But usually "polarity graph" means: V = Points of PG(2,q). E = {(P,Q) : Q lies on polar line of P}.
    # If P is on its polar line, P is adjacent to P? No, usually we consider distinct points.
    # Let's count edges for M=I over GF(5).
    # Points: 31.
    # For a point P, the polar line is x*x_P + y*y_P + z*z_P = 0.
    # This line contains 6 points.
    # One of them is P if P is absolute (x^2+y^2+z^2=0).
    # If P is absolute, it is adjacent to 5 other points.
    # If P is not absolute, it is adjacent to 6 points.
    # How many absolute points?
    # x^2+y^2+z^2=0 in PG(2,5).
    # Number of solutions to x^2+y^2+z^2=0 in GF(5)^3 \ {0}.
    # Squares in GF(5): 0, 1, 4.
    # Sum of 3 squares = 0.
    # Possible sums:
    # 0+0+0 = 0 (1 solution: 0,0,0 - excluded)
    # 1+4+0 = 5=0. (Permutations of (1,4,0) with signs).
    # Signs for 1: +/-1 (2 choices). Signs for 4: +/-2 (2 choices). Signs for 0: 1 choice.
    # Permutations of (1,4,0): 3! = 6.
    # Total non-zero solutions: 6 * 2 * 2 = 24.
    # So 24 points are absolute.
    # 31 - 24 = 7 points are not absolute.
    # Edges = (24 * 5 + 7 * 6) / 2 = (120 + 42) / 2 = 162 / 2 = 81.
    # This gives 81 edges. We need 91.
    
    # We need a polarity with more edges.
    # If we use a polarity where MORE points are non-absolute?
    # Or maybe a different form.
    # What if we use the form x*y + y*z + z*x = 0?
    # Or simply, since we know ex(31, C4)=90, and we need 91, we can start with the known 90-edge graph and add 1 edge, then optimize.
    # But we need to construct the 90-edge graph first.
    
    # The graph with 90 edges is the **Polarity Graph of PG(2,5)** with the **Unitary Polarity**.
    # As discussed, Unitary Polarity is defined over GF(25).
    # However, there is a graph on 31 vertices with 90 edges called the **Hoffman-Singleton** related? No.
    # It is the **Polarity Graph of PG(2,5)** associated with the **Hermitian** polarity?
    # Actually, for q=5, the maximum edges is 90.
    # The graph is 5-regular? 31*5/2 = 77.5. No.
    # It is not regular.
    
    # Let's try a different approach: **Randomized Construction with Strong Local Search**.
    # The current program uses random init.
    # We will use a **Greedy Construction** followed by **Simulated Annealing**.
    # Greedy: Start with empty graph. Add edges one by one that minimize the increase in C4 count.
    # This tends to build a sparse, C4-free core.
    # Then, use SA to fix the count to 91 and minimize C4.
    
    def greedy_construct(n, m):
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        # Precompute potential edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if len(edges) >= m:
                break
            # Check if adding (u,v) creates many C4s
            # C4s involving (u,v) are determined by common neighbors of u and v.
            # If we add (u,v), new C4s = number of common neighbors of u and v.
            # Wait, a C4 is u-w-x-v-u.
            # Adding (u,v) completes C4s for every pair (w,x) such that w is neighbor of u, x is neighbor of v, and (w,x) is an edge?
            # No, the C4 is u-w-x-v. The edge (w,x) must exist.
            # So new C4s = number of edges (w,x) where w in N(u), x in N(v).
            
            # Efficient check:
            # N(u) and N(v)
            # We want to minimize this count.
            
            # This greedy check is O(E) per edge, which is 31*30/2 * 91 ~ 40k ops. Very fast.
            # But we need to pick the BEST edge, not just the next random one.
            # So for each step, we evaluate all remaining candidates?
            # 91 steps * 465 candidates * 40k ops = 1.6 billion ops. Too slow for Python.
            
            # Alternative: Random Greedy.
            # Pick random edge, if it doesn't add too many C4s, add it.
            
            # Let's stick to Random Init + SA, but better SA.
            pass
        return edges

    # Strategy:
    # 1. Generate a random graph with 91 edges.
    # 2. Run Simulated Annealing.
    #    - Neighborhood: Swap one edge for another.
    #    - Cost: Number of C4s.
    #    - Acceptance: Metropolis criterion.
    #    - Cooling: Geometric.
    
    def get_c4_count(adj):
        A2 = adj @ adj
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        vals = A2[mask]
        return int(np.sum(vals * (vals - 1)) // 2)
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use a structured random init: Start with a random 5-regular-ish graph?
        # Or just random.
        random.seed(restart)
        adj = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        while len(edge_set) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edge_set.add((u, v))
        
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_score = get_c4_count(adj)
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edge_set)
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.99
        
        # Precompute list of edges for fast iteration
        current_edges = list(edge_set)
        
        # SA Loop
        steps = 0
        while T > T_min and time.time() < deadline:
            steps += 1
            # Pick a random edge to remove
            e_rm = random.choice(current_edges)
            u1, v1 = e_rm
            
            # Pick a random edge to add
            # To keep it connected-ish or just random? Random is fine.
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            e_add = (u2, v2)
            
            if e_add in edge_set:
                continue
                
            # Apply
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            new_score = get_c4_count(adj)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                edge_set.remove(e_rm)
                edge_set.add(e_add)
                current_edges.remove(e_rm)
                current_edges.append(e_add)
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edge_set)
            else:
                # Reject
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
            T *= cooling_rate
            
    if best_edges is None:
        # Fallback
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
        
    return [[u, v] for u, v in best_edges]

# To make it faster, we can optimize get_c4_count
# But 31x31 matmul is ~1000 flops. 
# 10000 steps * 1000 flops = 10M flops. Negligible.
# The bottleneck is Python overhead in the loop.
# We will run this.
