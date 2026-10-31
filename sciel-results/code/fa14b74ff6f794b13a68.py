import numpy as np
import time
from itertools import combinations
import sys

# Increase recursion limit for deep branch-and-bound searches
sys.setrecursionlimit(10000)

D = 11

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Generate all weight-4 integer vectors in R^11
    # These have norm squared R2 = 4.
    # Condition for kissing: ||v_i - v_j|| >= ||v_i|| = 2
    # ||v_i - v_j||^2 >= 4
    # 2*R2 - 2*v_i.v_j >= 4
    # 8 - 2*v_i.v_j >= 4
    # 4 >= 2*v_i.v_j
    # v_i.v_j <= 2
    # Since v_i, v_j are weight-4 vectors with entries in {-1, 1, 0},
    # their dot product is an integer.
    
    candidates = []
    # We need to choose 4 positions out of 11, and assign signs.
    # To avoid duplicates (v and -v are distinct vectors but symmetric in distance? 
    # No, v and -v have dot product -4, which is <= 2, so they are compatible.
    # However, we just need a set of vectors. 
    # Generating all 2^4 sign combinations for each position set covers all.
    
    for combo in combinations(range(D), 4):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[combo[0]] = s1
                        v[combo[1]] = s2
                        v[combo[2]] = s3
                        v[combo[3]] = s4
                        candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Compute conflict graph
    # Two vectors conflict if their dot product > 2.
    # G = candidates @ candidates.T
    # We only care about i != j.
    # diag is 4.
    
    # To save memory and time, we can compute the conflict matrix in chunks or just use int8/bool.
    # n_cand = C(11,4) * 16 = 330 * 16 = 5280.
    # 5280 x 5280 matrix of bools is ~27 MB. This is fine.
    
    # Compute dot products
    # We can do this in one go if memory allows.
    dots = candidates @ candidates.T
    
    # Conflict if dots[i, j] > 2 for i != j
    # Note: dots[i, i] = 4, which is > 2, but we ignore self-loops in MIS.
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Tomita Algorithm for Maximum Independent Set
    # The graph is the conflict graph. An independent set in the conflict graph
    # is a set of vertices with no edges between them, i.e., a set of vectors
    # where no pair conflicts, i.e., a valid kissing configuration.
    
    # We implement a recursive branch-and-bound.
    # State: 
    #   P: set of vertices available to be added (initially all vertices)
    #   X: set of vertices already chosen (initially empty)
    #   best_size: size of best solution found so far
    #   best_set: the set of vertices in the best solution
    
    # Optimization: 
    # Order vertices by degree (descending) for better pruning? 
    # Tomita uses a specific ordering. 
    # Let's use a simple recursive B&B with upper bound = |P|.
    # To improve, we can use a greedy coloring of the subgraph induced by P to get a tighter upper bound.
    # However, implementing full Tomita with coloring is complex. 
    # A simpler B&B with good vertex ordering might suffice for 5280 nodes if the graph is sparse enough.
    
    # Let's check sparsity.
    # Total pairs: 5280 * 5279 / 2 ~ 13.9 million.
    # Conflict if dot > 2.
    # Max dot is 4 (identical vectors, but we have unique vectors in list? 
    # Wait, my generation creates unique vectors. 
    # v and -v are distinct. dot(v, -v) = -4. Compatible.
    # v and v' with 3 common positions: dot = 3 or 1 or -1 or -3.
    # If 3 common with same signs: dot = 3. Conflict (3 > 2).
    # If 3 common with 2 same, 1 diff: dot = 1. Compatible.
    # If 4 common: dot = 4 (if same) or less. But vectors are unique, so no exact duplicates.
    # So conflicts arise mainly from vectors sharing 3 or 4 positions with similar signs.
    
    # Let's implement a basic Branch and Bound.
    # To make it fast, we need a good initial lower bound (greedy solution) and a good upper bound.
    
    # 1. Get a good initial solution using greedy
    #    Sort vertices by degree (ascending) and pick greedily? 
    #    Or just random greedy a few times.
    
    def greedy_mis():
        # Random greedy
        remaining = set(range(n_cand))
        selected = set()
        # To speed up, we can use a list and filter, but sets are O(1) remove.
        # However, checking compatibility requires checking against all selected.
        # For greedy, we can just iterate and check.
        
        # Better: Use the conflict matrix.
        # We want to pick a vertex, remove it and its neighbors.
        
        # Let's do a simple greedy: 
        # Order vertices by degree (highest first? No, usually lowest degree first for MIS? 
        # Actually, for MIS, picking high degree vertices early can prune more? 
        # Let's just use a random order or degree order.
        
        # Let's try degree ascending (pick isolated-looking vertices first).
        degrees = np.sum(conflict, axis=1)
        order = np.argsort(degrees)
        
        selected = []
        active = np.ones(n_cand, dtype=bool)
        
        for v in order:
            if not active[v]:
                continue
            selected.append(v)
            # Remove v and its neighbors from active
            # active[v] = False
            # active[conflict[v] == 1] = False
            
            # Vectorized update
            # But we need to be careful not to remove already processed vertices? 
            # No, active tracks what's still available.
            
            # Update active
            # We can use numpy bitwise_and with ~conflict_row
            # But conflict is int8.
            
            # Let's just loop for simplicity in greedy, n=5280 is small.
            # Actually, vectorized is faster.
            
            # Create a mask of neighbors to remove
            # neighbors = conflict[v] == 1
            # active[neighbors] = False
            # active[v] = False
            
            # We can precompute the neighbor lists? 
            # Or just do:
            # active &= ~(conflict[v] | (np.arange(n_cand) == v))
            # This creates a new array each time. 5280 * 5280 operations? No, 5280 ops per step.
            # 5280 steps * 5280 ops = 28M ops. Fast enough.
            
            # Let's use a list of neighbors for faster removal?
            # No, numpy is fine.
            
            # Optimization: 
            # We can maintain a list of active vertices and filter it.
            # But boolean mask is easier.
            
            pass # Placeholder, will implement properly below
            
        return selected
    
    # Let's implement a faster greedy using bitsets or just efficient numpy.
    # Actually, for the B&B, we need the graph structure.
    
    # Let's implement the Tomita algorithm properly.
    # Reference: Tomita et al., "The worst-case time complexity for enumerating maximal cliques".
    # Adapted for MIS (which is Clique in the complement graph).
    # Complement graph: edge if compatible (dot <= 2).
    # Let's work with the compatible graph directly for MIS? 
    # No, standard Tomita is for Max Clique. 
    # Max Independent Set in G = Max Clique in G_bar.
    
    # Let's build the adjacency matrix of the compatible graph (G_bar).
    # comp[i, j] = 1 if dots[i, j] <= 2 and i != j.
    comp = (dots <= 2).astype(np.int8)
    np.fill_diagonal(comp, 0)
    
    # Now we want Max Clique in this graph.
    # Tomita algorithm:
    # 1. Order vertices.
    # 2. Recursive search:
    #    expand(R, P):
    #       if P is empty:
    #           if |R| > best_size: update best
    #           return
    #       if |R| + |P| <= best_size: return (prune)
    #       
    #       # Coloring of P to get upper bound
    #       # Color P such that adjacent vertices have different colors.
    #       # The max color used is an upper bound on the clique size in P.
    #       # If |R| + max_color <= best_size: return
    #       
    #       # Pick a vertex v from P with highest color (or last in order)
    #       # expand(R U {v}, P ∩ N(v))
    #       # Remove v from P
    #       # Repeat for all v in P
    
    # Implementing coloring efficiently in Python is tricky.
    # However, for 5280 vertices, a simple greedy coloring might be slow if done at every node.
    # But we can use a simpler bound: |P|.
    # If we just use |P| as bound, it's less effective.
    
    # Let's try a simpler Branch and Bound with |P| bound first, with good ordering.
    # If it's too slow, we might not finish.
    # Given 150s limit, and 5280 nodes, simple B&B might struggle.
    
    # Alternative: Use a library? No, only standard + numpy/scipy.
    
    # Let's implement a reasonably optimized Tomita with greedy coloring.
    
    # Precompute adjacency lists for the compatible graph (for fast intersection)
    # adj[i] = list of j such that comp[i, j] == 1
    adj = [np.where(comp[i] == 1)[0] for i in range(n_cand)]
    
    # Sort vertices by degree (descending) for initial ordering? 
    # Tomita often uses degree ordering.
    degrees = np.array([len(adj[i]) for i in range(n_cand)])
    order = np.argsort(degues)[::-1] # Highest degree first
    
    # Remap indices to ordered indices? 
    # Or just use the order in the recursion.
    
    # Global best
    best_size = 0
    best_clique = []
    
    # Greedy initial solution
    def greedy_clique():
        # Simple greedy: pick highest degree, remove neighbors, repeat.
        P = set(range(n_cand))
        R = []
        while P:
            # Pick vertex with max degree in induced subgraph P? 
            # Approximate: pick max original degree in P.
            # To be fast, just pick first one? No, that's bad.
            # Let's pick max degree.
            # This is O(|P|) per step.
            
            # Optimization: just pick a random one? 
            # Let's do a few random restarts for greedy.
            pass
            
    # Let's do a simple random greedy to get a lower bound.
    for _ in range(100):
        P = list(range(n_cand))
        np.random.shuffle(P)
        R = []
        active = set(P)
        for v in P:
            if v not in active:
                continue
            R.append(v)
            # Remove v and its neighbors from active
            # neighbors = adj[v]
            # active -= set(adj[v])
            # active.remove(v)
            
            # Efficient removal:
            # We can just mark them as inactive in a boolean array?
            # But we are iterating over P.
            
            # Let's use a boolean array for active.
            # This is getting messy. Let's stick to the recursive B&B with |P| bound 
            # and hope the graph structure helps.
            
            # Actually, let's use a simpler heuristic: 
            # The kissing number in 11 is known to be between 504 and 506? 
            # Wait, kissing number k(11) is unknown exactly, but bounds are 504 <= k(11) <= 506?
            # Actually, Levenshtein bound gives 506.
            # So we are looking for a clique of size ~500 in a graph of 5280 nodes.
            
            # This is a dense graph? 
            # Number of compatible pairs:
            # Total pairs ~ 14M.
            # Conflicts are when dot > 2.
            # Most pairs have dot 0, 1, -1, etc.
            # So the compatible graph is VERY dense.
            # Max Clique in a dense graph is hard, but B&B might work if the upper bound is tight.
            # With |P| bound, the first branch goes deep.
            
            pass
            
    # Let's implement the Tomita algorithm with coloring.
    # It's the best bet for exact solution.
    
    # We need to represent P as a list of vertices.
    # And use a coloring array.
    
    # To speed up, we will use the "pivot" selection or just simple ordering.
    
    # Let's implement expand(R, P)
    # R: list of vertices in current clique
    # P: list of vertices available
    
    # We will use a global array for colors to avoid reallocating.
    
    # However, Python recursion and list operations might be slow.
    
    # Alternative Strategy:
    # Since the graph is dense, maybe we can find the MIS by removing conflicts?
    # No, that's the same.
    
    # Let's try a different approach: 
    # Use scipy.sparse to represent the graph and use a simple heuristic?
    # No, we need the exact size or close to it.
    
    # Let's go with the Tomita implementation.
    
    # To make it faster, we can use numpy for the coloring step?
    # Coloring is sequential.
    
    # Let's try to implement a simplified Tomita.
    
    # We'll use a global deadline check.
    
    def tomita_expand(R, P):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
            
        # Upper bound: |R| + |P|
        if len(R) + len(P) <= best_size:
            return
            
        # Coloring of P
        # We want to color P such that adjacent vertices have different colors.
        # We use a greedy coloring algorithm.
        # The max color assigned is an upper bound on the max clique in P.
        
        # To make coloring fast, we can use the order of P.
        # We will assign colors 1, 2, ...
        
        # We need to know if a vertex is adjacent to any vertex of the same color.
        # We can maintain a set of vertices for each color.
        
        # But P changes.
        
        # Let's use a simple greedy coloring on the subgraph induced by P.
        # Order P by degree (descending) for better coloring?
        # We already have global order.
        
        # Let's just iterate through P and assign the lowest possible color.
        
        # To do this efficiently, we need to check neighbors in P.
        
        # This is O(|P|^2) per node. Too slow.
        
        # Optimization: 
        # Use a bitset representation for P and colors?
        # Python integers can act as bitsets.
        
        # Let's represent P as a Python integer (bitset).
        # n_cand = 5280. Python int can handle this.
        
        # Precompute bitsets for adjacency?
        # adj_bitset[i] = integer with bits set for neighbors of i.
        
        # Then coloring:
        # For each vertex v in P (ordered):
        #   Find available colors.
        #   A color c is available if no neighbor of v in P has color c.
        
        # This is still complex.
        
        # Let's try a simpler B&B without coloring, but with a better bound?
        # Or just run the simple B&B and see if it finishes.
        # Given the dense graph, the search tree might be small if we prune well?
        # No, dense graphs are hard for Max Clique.
        
        # Wait, if the graph is dense, the MIS is small?
        # No, MIS in dense graph is small. Clique in dense graph is large.
        # We are finding Max Clique in the compatible graph.
        # Compatible graph is dense (most pairs are compatible).
        # So Max Clique should be large (~500).
        # B&B for large cliques in dense graphs is often efficient because the upper bound |P| drops quickly?
        # No, |P| stays large until we pick many vertices.
        
        # Let's try the simple B&B with |P| bound.
        
        # We will use a recursive function that takes P as a list.
        # To speed up, we sort P by degree.
        
        # Let's implement this.
        
        pass # Placeholder

    # Given the complexity and time limit, let's use a heuristic that is very fast:
    # 1. Greedy construction with random restarts.
    # 2. Local search (swap) to improve.
    
    # This is safer than a complex exact solver that might timeout.
    # The inspiration 1 (score 1.89) likely used a good heuristic.
    
    # Let's implement a robust heuristic.
    
    # 1. Generate candidates (done).
    # 2. Compute compatibility (done).
    # 3. Greedy MIS with multiple restarts.
    
    def greedy_mis_fast():
        # Use a random order.
        order = np.random.permutation(n_cand)
        selected = []
        active = np.ones(n_cand, dtype=bool)
        
        for v in order:
            if not active[v]:
                continue
            selected.append(v)
            # Remove v and neighbors
            # active[v] = False
            # active[adj[v]] = False
            
            # Vectorized:
            # We can precompute the "neighbor mask" for each v?
            # No, just use the conflict matrix.
            # active &= ~(conflict[v] | (np.arange(n_cand) == v))
            
            # This is O(N) per selection.
            # N=5280. 5280 selections * 5280 ops = 28M ops.
            # This is fast in numpy.
            
            # Let's optimize the mask creation.
            # conflict[v] is a row of the conflict matrix.
            # We need to set active[v] to False and active[neighbors] to False.
            
            # We can do:
            # active[v] = False
            # active[conflict[v] == 1] = False
            
            # But indexing with boolean array is slow?
            # Let's try.
            
            pass
            
    # Let's implement the heuristic properly.
    
    best_set = []
    best_size = 0
    
    # Precompute conflict rows as boolean arrays for faster indexing?
    # conflict_bool = (conflict == 1)
    
    # Let's do 1000 restarts of greedy.
    num_restarts = 500
    
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(n_cand)
        selected = []
        active = np.ones(n_cand, dtype=bool)
        
        for v in order:
            if not active[v]:
                continue
            
            selected.append(v)
            
            # Remove v and its neighbors
            # active[v] = False
            # neighbors = np.where(conflict[v] == 1)[0]
            # active[neighbors] = False
            
            # Faster:
            # active[v] = False
            # active &= ~(conflict[v] == 1) # This creates a new array
            
            # Let's use the conflict matrix directly.
            # conflict is int8.
            
            # We can precompute a list of neighbors for each v.
            # neighbors_list = [np.where(conflict[i] == 1)[0] for i in range(n_cand)]
            
            # Then:
            # active[v] = False
            # for u in neighbors_list[v]:
            #     active[u] = False
            
            # This is Python loop, might be slow if degree is high.
            # Degree in conflict graph:
            # How many vectors conflict with a given v?
            # v has 4 non-zeros.
            # Conflicts if dot > 2.
            # Only vectors sharing 3 or 4 positions with similar signs.
            # Number of such vectors is small.
            # So degree is low.
            # Python loop is fine.
            
            pass
            
    # Let's implement the neighbor lists.
    neighbors_list = [np.where(conflict[i] == 1)[0] for i in range(n_cand)]
    
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        order = np.random.permutation(n_cand)
        selected = []
        active = np.ones(n_cand, dtype=bool)
        
        for v in order:
            if not active[v]:
                continue
            
            selected.append(v)
            active[v] = False
            for u in neighbors_list[v]:
                active[u] = False
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    # Local search: try to add more vectors?
    # Or swap?
    
    # If best_size is 0, return a single vector.
    if best_size == 0:
        return candidates[:1]
        
    return candidates[best_set]

# Run construct
# Note: The actual code will be executed by the evaluator.

if __name__ == "__main__":
    # Test
    res = construct()
    print(res.shape)
