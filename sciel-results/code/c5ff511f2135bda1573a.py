# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    
    # Use finite field GF(25) construction
    # Represent elements as pairs (a, b) in GF(5)^2
    # Multiplication: (a,b)*(c,d) = (ac - bd, ad + bc) since 25 = 5^2 and -1 is non-square in GF(5)
    # Actually, -1 IS a square in GF(5) (2^2=4=-1), so use different irreducible polynomial
    # Use x^2 - 2 which is irreducible over GF(5) (no roots: 0->-2, 1->-1, 2->2, 3->7=2, 4->14=4)
    
    def gf25_mul(a, b):
        """Multiply two elements of GF(25) represented as (a0, a1) meaning a0 + a1*alpha where alpha^2 = 2"""
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + 2*a1*b1) % 5, (a0*b1 + a1*b0) % 5)
    
    def gf25_pow(a, e):
        if e == 0:
            return (1, 0)
        result = (1, 0)
        base = a
        while e > 0:
            if e & 1:
                result = gf25_mul(result, base)
            base = gf25_mul(base, base)
            e >>= 1
        return result
    
    # Generate all non-zero elements of GF(25)
    # Find a primitive element
    primitive = None
    for a0 in range(5):
        for a1 in range(5):
            if a0 == 0 and a1 == 0:
                continue
            elem = (a0, a1)
            # Check if primitive: should generate all 24 non-zero elements
            order = 1
            current = elem
            for _ in range(23):
                current = gf25_mul(current, elem)
                order += 1
                if current == (1, 0):
                    break
            if order == 24:
                primitive = elem
                break
        if primitive:
            break
    
    if primitive is None:
        # Fallback to random
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    # Generate all 24 non-zero elements using primitive element
    nonzero_elements = []
    current = (1, 0)
    for i in range(24):
        nonzero_elements.append(current)
        current = gf25_mul(current, primitive)
    
    # Map the 31 vertices: 0 is the "point at infinity", 1-24 are non-zero GF(25) elements, 25-30 are...
    # Actually, let's use a different approach: use the projective line over GF(25)
    # PG(1,25) has 26 points. We need 31, so we need to add 5 more.
    # Alternative: use a different construction.
    
    # Let me try a simpler approach: use the affine plane of order 5
    # The affine plane AG(2,5) has 25 points. Add 6 more points somehow?
    # This is getting complicated. Let me stick with a modified polarity graph approach
    # but use a better optimization strategy.
    
    # Actually, let me try constructing a graph using finite field characters
    # For GF(5), the quadratic residues are {1, 4} and non-residues are {2, 3}
    
    # New approach: Use the fact that n=31 is close to 25+6
    # Construct a graph with 25 vertices from AG(2,5) and 6 additional vertices
    
    # For now, let's use a different strategy: start with a random graph and optimize
    # But use a smarter initialization based on algebraic structure
    
    # Use the polarity graph of PG(2,5) as before, but with better optimization
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    if len(points) != 31:
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency for the full polarity graph
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    
    polarity_edges = len(edges)
    
    if polarity_edges < m:
        random.seed(42)
        existing = set(edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                edges.append(e)
                if len(edges) == m:
                    break
        return [[u, v] for u, v in edges]
    
    # Helper to count C4s in the current graph using numpy for speed
    def count_c4_fast(adj_mat):
        """Count C4s using matrix operations"""
        A = np.array(adj_mat, dtype=np.int32)
        # Number of C4s = sum over all pairs {u,v} of C(common_neighbors(u,v), 2)
        # where common_neighbors = A[u,:] dot A[v,:]
        # But we need to be careful: this counts each C4 multiple times
        # A C4 u-v-w-x-u is counted when we consider pairs (u,w) and (v,x)
        # Let's use: C4_count = (1/8) * sum_{u,v} (A^2)_{uv} * (A^2)_{uv} - ... 
        # Actually, simpler: for each pair (u,v), count common neighbors, then sum C(common,2)
        # This counts each C4 exactly 2 times (once for each pair of opposite vertices)
        
        c4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.sum(A[u] * A[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2  # Each C4 counted twice
    
    # Helper to compute the number of C4s an edge (u,v) belongs to
    def count_c4s_for_edge(adj_mat, u, v):
        count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[u][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[v][x] and adj_mat[w][x]:
                        count += 1
        return count

    # Helper to compute the number of C4s created by adding edge (u,v)
    def count_c4s_created_by_adding(adj_mat, u, v):
        count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[v][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[u][x] and adj_mat[w][x]:
                        count += 1
        return count

    # Initialize graph: Remove edges from polarity graph to get 91 edges
    current_adj = [row[:] for row in adj]
    current_edges = set(edges)
    
    # Remove edges to reach exactly m edges
    to_remove = polarity_edges - m
    for _ in range(to_remove):
        best_edge = None
        best_count = -1
        for e in current_edges:
            u, v = e
            count = count_c4s_for_edge(current_adj, u, v)
            if count > best_count:
                best_count = count
                best_edge = e
        if best_edge:
            current_edges.remove(best_edge)
            u, v = best_edge
            current_adj[u][v] = current_adj[v][u] = False
    
    current_c4 = count_c4_fast(current_adj)
    
    # List of all possible non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated annealing for better exploration
    best_c4 = current_c4
    best_edges = set(current_edges)
    best_adj = [row[:] for row in current_adj]
    
    temperature = 50.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Calculate C4 counts for all current edges
        edge_c4_counts = {}
        for e in current_edges:
            u, v = e
            edge_c4_counts[e] = count_c4s_for_edge(current_adj, u, v)
        
        # Calculate C4 creation counts for all non-edges
        non_edges = [e for e in all_pairs if e not in current_edges]
        non_edge_c4_counts = {}
        for e in non_edges:
            u, v = e
            non_edge_c4_counts[e] = count_c4s_created_by_adding(current_adj, u, v)
        
        # Try a random swap
        e_out = random.choice(list(current_edges))
        e_in = random.choice(non_edges)
        
        u_out, v_out = e_out
        u_in, v_in = e_in
        
        # Apply swap
        current_adj[u_out][v_out] = False
        current_adj[v_out][u_out] = False
        current_adj[u_in][v_in] = True
        current_adj[v_in][u_in] = True
        
        new_c4 = count_c4_fast(current_adj)
        delta = new_c4 - current_c4
        
        # Accept if better or with probability exp(-delta/temperature)
        if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
            current_edges.remove(e_out)
            current_edges.add(e_in)
            current_c4 = new_c4
            
            # Update best
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(current_edges)
                best_adj = [row[:] for row in current_adj]
        else:
            # Revert
            current_adj[u_out][v_out] = True
            current_adj[v_out][u_out] = True
            current_adj[u_in][v_in] = False
            current_adj[v_in][u_in] = False
        
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
    
    # Use the best found solution
    final_edges = list(best_edges)
    
    # Verify
    if len(final_edges) != m:
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        final_edges = all_possible[:m]
    
    result = [[u, v] for u, v in final_edges]
    return result

# EVOLVE-BLOCK-END
