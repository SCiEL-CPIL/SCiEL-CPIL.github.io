import numpy as np
import random
import time
import sys

def construct():
    # Deadline: 120s. Use 110s for safety.
    deadline = time.time() + 110
    
    # Try n from 17 down to 1
    for n in range(17, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Run Simulated Annealing for this n
        best_coloring, best_cost = solve_sa(n, time_limit=5.0)
        
        # If we found a solution with cost 0, verify it exactly
        if best_cost == 0 and best_coloring is not None:
            if verify_no_c6(n, best_coloring):
                return {'n': n, 'colors': best_coloring}
        
        # If we didn't find 0, but we have time, we might try again or move to smaller n
        # Given the difficulty, moving to smaller n is safer for guaranteeing a score > seed (9)
        if n < 10:
             # We should have found n=9 easily
             break

    # Fallback: Return seed-like result if we failed (should not happen for n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def get_edge_indices(n):
    """Generate list of (u, v) pairs for all edges in K_n in lexicographic order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def color_to_matrix(n, colors):
    """Convert flat color list to 4 adjacency matrices."""
    m = n * (n - 1) // 2
    # Initialize 4 matrices
    mats = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    edges = get_edge_indices(n)
    for k, (u, v) in enumerate(edges):
        c = colors[k]
        mats[c][u, v] = 1
        mats[c][v, u] = 1
    return mats

def count_walks_len_5(mat, u, v):
    """Count number of walks of length 5 from u to v in graph defined by mat."""
    # vec[i] = number of walks of length 'len' from u to i
    vec = np.zeros(len(mat), dtype=np.int64)
    vec[u] = 1
    
    for _ in range(5):
        vec = mat @ vec
        # vec is now walks of length 1, then 2... then 5
        # Note: mat is 0/1 int8, vec grows. int64 is safe for n=17.
        
    return int(vec[v])

def compute_cost(mats):
    """Approximate cost: Sum of traces of A_c^6.
    This counts closed walks of length 6.
    """
    cost = 0
    for mat in mats:
        # Trace(A^6) = sum of diagonal elements of A^6
        # We can compute A^6 or just sum (A^5) @ A diagonal?
        # Let's just do matrix multiplication for simplicity, n=17 is small.
        # A2 = A@A, A4 = A2@A2, A6 = A4@A2
        # Or just:
        # cost_c = np.trace(np.linalg.matrix_power(mat, 6))
        # But matrix_power might be slow if called often.
        # Let's compute A5 once per matrix and use it?
        # Actually, for delta calculation in SA, we use count_walks_len_5.
        # For total cost, let's just compute it directly to avoid drift.
        A5 = mat @ mat @ mat @ mat @ mat
        # Trace(A^6) = sum_i (A^6)_{ii} = sum_i (A^5 @ A)_{ii} = sum_i (A^5)_{i,?} ...
        # Actually (A^6)_{ii} = sum_j (A^5)_{ij} A_{ji} = sum_j (A^5)_{ij} A_{ij}
        # So cost_c = sum_{i,j} (A^5)_{ij} * A_{ij}
        # This is efficient: element-wise multiply and sum.
        cost_c = np.sum(A5 * mat)
        cost += cost_c
    return cost

def solve_sa(n, time_limit):
    m = n * (n - 1) // 4
    # 4 colors
    edges = get_edge_indices(n)
    
    # Initial random coloring
    colors = [random.randint(0, 3) for _ in range(m)]
    mats = color_to_matrix(n, colors)
    current_cost = compute_cost(mats)
    
    best_colors = colors.copy()
    best_cost = current_cost
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.99
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        idx = random.randint(0, m - 1)
        u, v = edges[idx]
        c_old = colors[idx]
        
        # Pick new color
        possible_new = [c for c in range(4) if c != c_old]
        c_new = random.choice(possible_new)
        
        # Calculate delta cost
        # Cost contribution of edge (u,v) in color c_old:
        # It is part of closed walks of length 6.
        # Number of walks of length 5 between u and v in G_{c_old} is W_old.
        # Each such walk + edge (u,v) forms a closed walk of length 6.
        # There are 2 directions? No, walk u->v then edge v->u? Or u->v is the edge?
        # In undirected graph, walk u->...->v of len 5. Adding edge (u,v) creates cycle u->...->v->u.
        # This is a closed walk of length 6.
        # Does it count as 1 or 2 in Trace?
        # Trace counts closed walks starting and ending at each node.
        # A specific cycle u-...-v-u is counted 2 times in Trace? (Start at u, start at v? No, Trace sums over all starts).
        # Actually, let's just use the property:
        # Cost = sum_c sum_{i,j} (A_c^5)_{ij} A_{ij}
        # Delta for removing edge (u,v) from c_old:
        # Term (u,v) and (v,u) disappear from A.
        # Change in sum = - ( (A^5)_{uv} * 1 + (A^5)_{vu} * 1 ) = -2 * (A^5)_{uv} (since symmetric)
        # Wait, A^5 is symmetric.
        # So delta_remove = -2 * count_walks_len_5(mats[c_old], u, v)
        
        # Change for adding edge (u,v) to c_new:
        # delta_add = 2 * count_walks_len_5(mats[c_new], u, v)
        
        # BUT: Adding edge to c_new changes A_{c_new}, which changes A_{c_new}^5 for ALL pairs, not just (u,v).
        # So the approximation delta = 2*W_new - 2*W_old is only valid for the DIRECT term.
        # The indirect terms (changes in other walks) are ignored.
        # This is a first-order approximation. It works for SA to guide the search.
        
        w_old = count_walks_len_5(mats[c_old], u, v)
        w_new = count_walks_len_5(mats[c_new], u, v)
        
        delta_cost = 2 * (w_new - w_old)
        
        # Acceptance
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            # Accept move
            # Update matrices
            mats[c_old][u, v] = 0
            mats[c_old][v, u] = 0
            mats[c_new][u, v] = 1
            mats[c_new][v, u] = 1
            colors[idx] = c_new
            
            # Recalculate exact cost periodically or use delta?
            # Using delta accumulates error. Let's recalculate exact cost every 100 steps or if delta is large.
            # For now, let's just use the approximate current_cost to track best, 
            # but verify with exact cost when we think we are good.
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Periodic exact recalculation to correct drift
        if idx % 50 == 0:
            current_cost = compute_cost(mats)
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()

    # Final verification of best found
    if best_cost > 0:
         # It's possible best_cost is an approximation. 
         # We return best_colors. The caller will verify.
         pass
         
    return best_colors, best_cost

def verify_no_c6(n, colors):
    """
    Check if any color class contains a C6.
    n <= 20.
    """
    m = n * (n - 1) // 2
    # Create adjacency matrices for each color
    adj = [set() for _ in range(4)]
    edges = get_edge_indices(n)
    for k, (u, v) in enumerate(edges):
        c = colors[k]
        adj[c].add((u, v))
        adj[c].add((v, u))
    
    # Check each color
    for c in range(4):
        if has_c6(n, adj[c]):
            return False
    return True

def has_c6(n, adj_set):
    """Check if graph (n, adj_set) contains a C6."""
    # Iterate through all combinations of 6 vertices
    # Since n is small (<=20), this is feasible.
    import itertools
    vertices = range(n)
    # Pre-compute adjacency matrix for faster lookup
    adj_mat = np.zeros((n, n), dtype=np.int8)
    for u, v in adj_set:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
        
    # There are 60 distinct ways to form a C6 on 6 labeled vertices (up to rotation/reflection)
    # Vertices 0,1,2,3,4,5 in a cycle: 0-1-2-3-4-5-0
    # We can just check all permutations? No, too many.
    # We can check all sets of 6 vertices. For each set, check if the induced subgraph contains a C6.
    # Induced subgraph has 15 edges.
    # A C6 uses 6 edges.
    # We can iterate through all 6! / 12 = 60 unique cycles on 6 labeled nodes.
    
    # Define the 60 unique edge sets for a C6 on nodes 0..5
    # Relative to the sorted tuple of 6 vertices
    c6_edge_sets = []
    # One representative: 0-1-2-3-4-5-0
    # We generate all permutations of 1..5 and check if we've seen this set before.
    seen = set()
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        cycle = [0] + list(perm)
        edges_in_cycle = set()
        for i in range(6):
            u, v = cycle[i], cycle[(i+1)%6]
            edge = tuple(sorted((u, v)))
            edges_in_cycle.add(edge)
        # Normalize the edge set to handle symmetries (rotation/reflection of the cycle itself is handled by permutations, 
        # but the set of edges is unique for each structural cycle type on fixed labels? 
        # Actually, different permutations of 1..5 can yield the same set of edges on 0..5.
        # Example: 0-1-2-3-4-5-0 and 0-5-4-3-2-1-0 are the same set of edges.
        # So we add the frozenset of edges to 'seen'.
        edge_tuple = tuple(sorted(edges_in_cycle))
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            c6_edge_sets.append(edge_tuple)
            
    # There should be 60 unique sets? 
    # Number of Hamiltonian cycles in K6 is (6-1)!/2 = 60.
    # Yes, 60 unique edge sets.
    
    for combo in itertools.combinations(vertices, 6):
        # Map combo indices to 0..5
        mapping = {v: i for i, v in enumerate(combo)}
        
        # Check if any of the 60 patterns matches
        for edge_set in c6_edge_sets:
            # Check if all edges in this pattern exist in the graph
            match = True
            for u_rel, v_rel in edge_set:
                u_abs = combo[u_rel]
                v_abs = combo[v_rel]
                if adj_mat[u_abs, v_abs] == 0:
                    match = False
                    break
            if match:
                return True
    return False

# Main entry
if __name__ == "__main__":
    # For debugging
    res = construct()
    print(res['n'])
