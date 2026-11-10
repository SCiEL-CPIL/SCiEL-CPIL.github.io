import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))
EDGE_INDEX = {e: i for i, e in enumerate(ALL_PAIRS)}
NUM_EDGES = len(ALL_PAIRS)

# Precompute Q3 masks for K8
def generate_q3_masks():
    # Standard Q3 edges
    # Vertices 0..7
    # 0-1, 1-3, 3-2, 2-0
    # 4-5, 5-7, 7-6, 6-4
    # 0-4, 1-5, 3-7, 2-6
    std_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (3, 7), (2, 6)
    ]
    
    masks = set()
    for perm in itertools.permutations(range(8)):
        # perm[i] is the new label for old vertex i
        # Or rather, map old vertex v to new vertex perm[v]?
        # Let's say we map old vertex i to perm[i].
        # Then edge (u, v) in std becomes (perm[u], perm[v]).
        new_edges = []
        for u, v in std_edges:
            nu, nv = perm[u], perm[v]
            if nu > nv:
                nu, nv = nv, nu
            new_edges.append((nu, nv))
        # Convert to frozenset for hashing
        masks.add(frozenset(new_edges))
        
    return list(masks)

Q3_MASKS = generate_q3_masks()

def find_q3_in_graph(edges):
    """
    edges: set of tuples (u, v)
    Returns a set of 12 edges forming a Q3, or None.
    """
    if len(edges) < 12:
        return None
        
    # Try random subsets of 8 vertices
    # If the graph is dense, this should find one quickly.
    # If the graph is Q3-free, this will fail.
    # We need a robust way to declare "not found".
    # Given the density, random sampling is good.
    
    vertices = list(range(N))
    rng = random.Random(42)
    
    for _ in range(2000): # Limit checks
        sample_vertices = rng.sample(vertices, 8)
        # Map sample vertices to 0..7
        index_map = {v: i for i, v in enumerate(sample_vertices)}
        
        # Check each Q3 mask
        for mask in Q3_MASKS:
            # Check if all edges in mask are present in the graph
            # Mask edges are (u, v) where u,v are in 0..7
            # We need to check if (sample_vertices[u], sample_vertices[v]) is in edges
            # Optimization: Precompute the set of edges in the sample induced subgraph
            # But doing this for every mask is expensive.
            
            # Better: Check edges one by one.
            found = True
            for u, v in mask:
                eu, ev = sample_vertices[u], sample_vertices[v]
                if eu > ev:
                    eu, ev = ev, eu
                if (eu, ev) not in edges:
                    found = False
                    break
            if found:
                # Construct the actual edges
                q3_edges = set()
                for u, v in mask:
                    eu, ev = sample_vertices[u], sample_vertices[v]
                    if eu > ev:
                        eu, ev = ev, eu
                    q3_edges.add((eu, ev))
                return q3_edges
                
    return None

def construct():
    start = time.time()
    deadline = start + 110.0

    # Fallback greedy
    def greedy_construct():
        rng = random.Random(12345)
        best_edges = None
        best_count = -1
        
        def time_left():
            return deadline - time.time()

        trial = 0
        while time_left() > 5.0:
            trial += 1
            order = ALL_PAIRS[:]
            rng.shuffle(order)

            adj = [0] * N
            edges = []
            count = 0
            
            for e in order:
                if time_left() < 1.0:
                    break
                u, v = e
                if (adj[u] >> v) & 1:
                    continue
                if not _creates_cube(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    count += 1

            if count > best_count:
                best_count = count
                best_edges = edges

        if best_edges is None:
            best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
        return best_edges

    # Try MILP Cutting Plane
    try:
        # Objective: Maximize sum x_e -> Minimize -sum x_e
        c = -np.ones(NUM_EDGES)
        
        # Bounds
        bounds = Bounds(lb=0, ub=1)
        
        # Constraints
        # Start with no constraints
        A = []
        lb = []
        ub = []
        
        # We need to add constraints dynamically.
        # scipy.optimize.milp requires A to be a 2D array.
        # We will rebuild it each time.
        
        current_constraints = [] # List of (mask_indices, 11)
        
        best_solution = None
        best_edges_count = -1
        
        while time.time() < deadline - 10.0:
            # Build A matrix
            if current_constraints:
                A_list = []
                lb_list = []
                ub_list = []
                for mask_indices, bound in current_constraints:
                    row = np.zeros(NUM_EDGES)
                    for idx in mask_indices:
                        row[idx] = 1.0
                    A_list.append(row)
                    lb_list.append(0) # Actually we want sum <= 11, so sum >= -inf, sum <= 11
                    ub_list.append(bound)
                
                A = np.vstack(A_list)
                lb = np.array(lb_list)
                ub = np.array(ub_list)
                constraints = LinearConstraint(A, lb, ub)
            else:
                constraints = LinearConstraint(A=np.zeros((0, NUM_EDGES)), lb=[], ub=[])
            
            # Solve
            res = milp(c, constraints=constraints, integrality=np.ones(NUM_EDGES), bounds=bounds, options={"time_limit": 10})
            
            if res.status != 0:
                break # Solver failed or timed out
            
            x = res.x
            edges_set = set()
            for i, val in enumerate(x):
                if val > 0.5:
                    edges_set.add(ALL_PAIRS[i])
            
            if len(edges_set) > best_edges_count:
                best_edges_count = len(edges_set)
                best_solution = edges_set
                
            # Check for Q3
            q3 = find_q3_in_graph(edges_set)
            if q3 is None:
                # Found Q3-free graph
                # Convert to list of lists
                return [[u, v] for u, v in best_solution]
            else:
                # Add constraint
                # Get indices of edges in q3
                mask_indices = [EDGE_INDEX[e] for e in q3]
                current_constraints.append((mask_indices, 11))
                
    except Exception as e:
        pass

    # Fallback
    return greedy_construct()

def _creates_cube(u, v, adj):
    """Return True if adding edge (u,v) to adjacency `adj` (list of int bitsets)
    creates a 3-cube (Q3) as a subgraph.
    """
    Nu = adj[u] & ~(1 << v)
    Nv = adj[v] & ~(1 << u)
    
    # Get lists of neighbors
    L_u = [i for i in range(N) if (Nu >> i) & 1]
    L_v = [i for i in range(N) if (Nv >> i) & 1]
    
    if len(L_u) < 2 or len(L_v) < 2:
        return False
        
    for i in range(len(L_u)):
        x = L_u[i]
        for j in range(i+1, len(L_u)):
            w = L_u[j]
            # We need y in N(x) \cap N(v) and z in N(w) \cap N(v)
            # such that y != z and y,z distinct from x,w
            mask_y = adj[x] & Nv & ~(1 << x) & ~(1 << w)
            mask_z = adj[w] & Nv & ~(1 << x) & ~(1 << w)
            
            if not mask_y or not mask_z:
                continue
                
            # Check if there exist y in mask_y and z in mask_z with y != z
            # This fails only if mask_y == mask_z and it is a single bit
            if mask_y == mask_z and (mask_y & (mask_y - 1)) == 0:
                continue
                
            return True
    return False
