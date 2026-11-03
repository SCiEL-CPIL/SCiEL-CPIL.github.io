import time
import numpy as np
from itertools import combinations, product
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate candidate vectors
    # We look for nonzero integer vectors v in Z^11
    # Condition: min pairwise dist >= max norm
    # Let M = max norm. We require ||v_i|| <= M and ||v_i - v_j|| >= M.
    # Note: If we normalize by M, we get unit vectors with dist >= 1.
    # Standard kissing number problem usually implies equal radii, so equal norms.
    # However, the task says "minimum pairwise distance is at least the maximum vector norm".
    # This allows vectors of different norms, but to maximize count, we usually want them close to max norm.
    # Actually, if we have a set of vectors, let M = max ||v_i||.
    # We need ||v_i - v_j|| >= M for all i != j.
    # And ||v_i|| <= M.
    
    # Strategy: Enumerate all integer vectors with norm squared <= MAX_NORM_SQ.
    # What is a good MAX_NORM_SQ?
    # For D=11, known kissing number is likely around 55-60? (Actually K(11) is not known exactly, bounds are 55 <= K(11) <= 66? No, K(10)=45, K(12)=92? Wait.
    # K(1)=2, K(2)=6, K(3)=12, K(4)=24, K(5)=40, K(6)=72, K(7)=126, K(8)=240, K(9)=460, K(10)=870? No.
    # Let's check small dimensions.
    # In R^11, the best known lower bound is often constructed from lattice points.
    # The Leech lattice is in R^24.
    # In R^11, we can use the construction from the E8 lattice + something?
    # Actually, let's just search for integer vectors.
    # If we use vectors with norm squared = 2 (roots), they are orthogonal or at 60/120 deg?
    # ||v||^2 = 2. v_i in {1, -1, 0}. Two non-zeros.
    # There are 11*2 (single 1/-1 at pos) -> norm 1.
    # There are 11*10*2 (two non-zeros) -> norm 2.
    # If we take all norm 2 vectors, do they satisfy the condition?
    # Max norm M = sqrt(2).
    # Dist between two norm 2 vectors:
    # If supports disjoint: dist^2 = 4. 4 >= 2. OK.
    # If supports intersect at 1: dist^2 = 1 + 1 + 1 + 1 = 4? No.
    # v1 = (1, 1, 0...), v2 = (1, -1, 0...). dist^2 = 0 + 4 = 4.
    # v1 = (1, 1, 0...), v2 = (1, 0, 1...). dist^2 = 0 + 1 + 1 = 2. 2 >= 2. OK.
    # So ALL norm 2 vectors form a valid set?
    # Count: 11 * 10 * 2 = 220 vectors.
    # Wait, is M = sqrt(2)? Yes, all have norm sqrt(2).
    # Min dist is sqrt(2).
    # So k=220 is a valid solution?
    # Let's verify.
    # v1 = e1 + e2. v2 = e1 + e3. dist^2 = (e2-e3)^2 = 2. Norm^2 = 2.
    # v1 = e1 + e2. v2 = -(e1 + e2). dist^2 = (2e1+2e2)^2 = 8.
    # v1 = e1 + e2. v3 = -e1 + e2. dist^2 = (2e2)^2 = 4.
    # So yes, the set of all vectors with exactly two non-zero entries (each +/-1) is a valid kissing configuration?
    # The task asks for "nonzero integer vectors... whose minimum pairwise distance is at least the maximum vector norm".
    # If I return all 220 such vectors, the max norm is sqrt(2). The min dist is sqrt(2).
    # Condition: sqrt(2) >= sqrt(2). True.
    # So k=220 is a lower bound.
    
    # Can we do better?
    # What if we include norm 1 vectors?
    # Max norm becomes sqrt(2).
    # Dist between norm 1 (e1) and norm 2 (e1+e2): dist^2 = 1. 1 < 2. Invalid.
    # So we cannot mix norms if the larger norm is too big relative to the distance.
    # Specifically, if we include a vector v with ||v||=M, any other vector u must satisfy ||u-v|| >= M.
    # If u is small, ||u-v|| approx ||v|| = M. It might work if u is orthogonal to v.
    # But for v=e1+e2, u=e1 is not orthogonal. dist^2 = 1 < 2.
    
    # What if we use norm squared = 4 vectors?
    # e.g. 2*e1. Norm 2.
    # Dist to e1+e2: (2e1 - (e1+e2))^2 = (e1-e2)^2 = 2.
    # Max norm 2. 2 < 4. Invalid.
    
    # It seems sticking to a single norm shell is best for maximizing density in this "integer" constraint.
    # The best shell is likely norm^2=2 (k=220) or maybe norm^2=3?
    # Vectors with norm^2=3: three non-zeros, e.g. 1,1,1.
    # Count: 11*10*9*2^3 / 6? No, order matters for positions, signs matter.
    # Number of supports of size 3: C(11,3) = 165.
    # Signs: 2^3 = 8.
    # Total 165 * 8 = 1320 vectors.
    # Check validity for norm^2=3 shell.
    # Max norm M = sqrt(3).
    # Min dist between two vectors in this shell?
    # v1 = (1,1,1,0...), v2 = (1,1,-1,0...). dist^2 = 4. 4 >= 3. OK.
    # v1 = (1,1,1,0...), v2 = (1,1,0,1...). dist^2 = 1+1 = 2. 2 < 3. INVALID.
    # So we cannot take ALL norm^2=3 vectors. We must solve MIS on the conflict graph of this shell.
    
    # Similarly for norm^2=2 shell, we took ALL. Is it possible we missed some?
    # We established ALL norm^2=2 vectors are valid. So k=220 is a hard baseline from that shell.
    
    # Let's try to beat 220 using the norm^2=3 shell with MIS.
    # Or maybe a mix? No, mixing usually hurts due to the "max norm" constraint forcing the threshold up.
    # If we mix norm^2=2 and norm^2=3, M=sqrt(3).
    # Dist between norm^2=2 vector (e1+e2) and norm^2=3 vector (e1+e2+e3) is 1 (vector e3). 1 < 3. Invalid.
    # So they must be "far" apart.
    
    # Hypothesis: The optimal solution lies within a single shell of norm^2 = w.
    # We will generate all integer vectors with norm^2 = w for w in {2, 3, 4, 5, 6}.
    # Build conflict graph: Edge if dist^2 < w (since M^2 = w).
    # Find MIS.
    
    # Optimization:
    # For w=2, MIS is trivial (all vertices). k=220.
    # For w=3, we need MIS.
    # For w=4, vectors like (2,0,..) or (1,1,1,1..).
    
    # Let's implement the MIS solver.
    # Graph size for w=3: 1320 nodes.
    # Graph size for w=4: 
    # Supports of size 4: C(11,4)=330, signs 16 -> 5280.
    # Supports of size 2 with 2,0,..: 11*10*2 (pos) * 1 (sign of 2? no, 2 and -2) -> 220? 
    # Wait, integer vectors with norm^2=4:
    # Type A: one entry 2 or -2. 11 * 2 = 22.
    # Type B: two entries 1 or -1. 11*10*2^2 / 2? No, (1,1) and (1,-1) are distinct.
    # Pairs of indices: 55. Signs: 4. Total 220.
    # Type C: four entries 1 or -1. C(11,4)*16 = 330*16 = 5280.
    # Total ~5500 nodes.
    
    # We will iterate w from 2 to 6, find best MIS, and return the best vector set.
    
    def generate_vectors_for_norm_sq(w):
        vecs = []
        # We need to generate all integer vectors v in Z^11 such that sum(v_i^2) == w.
        # This is a bounded integer partition problem.
        # Since w is small, we can iterate over possible supports and values.
        
        # Possible values for entries:
        # If w=2: entries in {1, -1, 0}. Max 2 non-zeros.
        # If w=3: entries in {1, -1, 0}. Max 3 non-zeros.
        # If w=4: entries in {1, -1, 2, -2, 0}.
        # If w=5: entries in {1, -1, 2, -2, 0}. (1+4)
        # If w=6: entries in {1, -1, 2, -2, 0}. (1+1+4 or 1+5(no) or 2+2+2? 2^2+2^2+2^2=12 no. 1+1+1+1+1+1=6)
        
        # Generic generator:
        # Recursively assign values to indices 0..D-1 such that sum of squares == w.
        
        def gen(idx, remaining_sq, current_vec):
            if idx == D:
                if remaining_sq == 0:
                    if any(x != 0 for x in current_vec):
                        vecs.append(tuple(current_vec))
                return
            
            # Pruning: if remaining_sq < 0, return
            if remaining_sq < 0:
                return
            
            # Try all possible values for this index
            # Max value is sqrt(remaining_sq)
            max_val = int(np.sqrt(remaining_sq))
            
            # Symmetry breaking? No, we need all vectors.
            # But we can optimize by noting that if we put 0, we move to next idx.
            
            for v in range(-max_val, max_val + 1):
                new_remaining = remaining_sq - v*v
                # Optimization: if we choose v, the remaining indices must sum to new_remaining.
                # Also, if idx > 0 and v == 0 and current_vec[idx-1] == 0, we might be generating permutations?
                # No, indices are distinct positions.
                
                # Heuristic: To avoid generating duplicates or to speed up, we can enforce an order on non-zero entries?
                # No, (1,0) and (0,1) are different vectors.
                
                # Just iterate.
                current_vec.append(v)
                gen(idx + 1, new_remaining, current_vec)
                current_vec.pop()
        
        gen(0, w, [])
        
        # Remove duplicates? The generator produces unique tuples because indices are fixed.
        # Convert to numpy array
        if not vecs:
            return np.array([]).reshape(0, D)
        return np.array(vecs, dtype=np.int64)
    
    def solve_mis(adj_matrix, n):
        """
        Tomita's algorithm for Maximum Independent Set.
        adj_matrix: numpy bool array (n, n) where True means edge (conflict).
        We want max set of nodes with NO edges between them.
        This is equivalent to Max Clique in the complement graph.
        Or we can run Max Independent Set directly.
        
        Let's implement Max Independent Set on G.
        G has edge (u,v) if dist^2(u,v) < w.
        Independent Set in G = Clique in Complement(G).
        
        Let's build the complement graph adjacency for the clique solver.
        """
        # Complement graph: edge if NOT conflict.
        # comp_adj[i][j] = not adj_matrix[i][j] and i != j
        
        # Tomita's algorithm works on a graph to find Max Clique.
        # We will find Max Clique in the complement graph.
        
        # Build complement adjacency list for speed?
        # Bitsets are best.
        
        # n can be up to ~5500.
        # Bitset of 5500 bits is ~700 bytes.
        # Array of 5500 bitsets is ~3.8 MB. Very fast in Python if using integers?
        # Python integers are arbitrary precision.
        
        # Let's use Python integers as bitsets.
        # adj_bits[i] is an integer where bit j is 1 if i and j are connected in the CLIQUE graph (i.e., compatible).
        
        # First, compute compatibility.
        # We have the vectors.
        # dist^2 < w means conflict.
        # dist^2 >= w means compatible (edge in clique graph).
        
        # We pass adj_matrix (conflict) to this function?
        # Better to pass vectors and w, and compute bits inside.
        pass # Placeholder
    
    # Re-implementing the core logic to be self-contained and fast.
    
    best_k = 0
    best_vecs = None
    
    # Try different shells
    for w in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 20:
            break
            
        vecs = generate_vectors_for_norm_sq(w)
        n = len(vecs)
        if n == 0:
            continue
            
        # If n is small enough, we can use a simple greedy or exact solver.
        # For w=2, n=220. We know it's 220.
        # For w=3, n=1320.
        # For w=4, n~5500.
        
        # Build Conflict Matrix or Compatibility Bitsets.
        # Let's build Compatibility Bitsets for the Clique Solver.
        # Node i is compatible with Node j if dist^2(i,j) >= w.
        
        # Optimization: Vectorized distance calculation.
        # For n=5500, n^2 = 30M. 
        # 30M dot products of 11-dim vectors.
        # Using NumPy:
        # (vecs @ vecs.T) gives dot products.
        # dist^2 = ||v_i||^2 + ||v_j||^2 - 2 dot(v_i, v_j)
        # Since all have norm^2 = w, dist^2 = 2w - 2 dot(v_i, v_j).
        # Condition: 2w - 2 dot >= w  =>  w >= 2 dot  =>  dot <= w/2.
        
        dots = vecs @ vecs.T
        # Compatibility: dots[i][j] <= w/2
        # Note: dots[i][i] = w. w <= w/2 is False (for w>0). So diagonal is False (no self-loop in clique graph? 
        # Actually, in Clique problem, we don't pick self-loops. We pick sets of distinct nodes.
        # The diagonal should be False for the adjacency matrix of the graph we are finding cliques in.
        
        # Threshold
        thresh = w / 2.0
        # Compatible if dot <= thresh
        comp_mask = dots <= thresh
        # Ensure diagonal is False
        np.fill_diagonal(comp_mask, False)
        
        # Convert to bitsets (Python ints)
        # For each i, adj_bits[i] has bit j set if comp_mask[i][j] is True.
        adj_bits = [0] * n
        # This loop is O(n^2 / word_size). In Python, bit manipulation on large ints is fast.
        # But constructing them from a numpy array is the bottleneck.
        
        # Trick: Use numpy to pack bits?
        # Or just loop if n is not huge.
        # For n=5500, 5500 iterations. Inside, we need to set bits.
        # We can iterate over rows.
        
        # Faster: 
        # Create an array of indices for True values in each row?
        # Or use the fact that we just need the int.
        
        # Let's try a vectorized bit-packing if possible, or just loop.
        # Looping 5500 times is fine.
        
        for i in range(n):
            # Get indices of compatible nodes
            # comp_mask[i] is a bool array of size n.
            # We want an int where bit j is 1 if comp_mask[i][j].
            # We can do this by:
            # 1. Find non-zero indices
            # 2. Sum 1 << j
            
            # This is slow in Python loop.
            # Alternative: 
            # Since we only need to run MIS, and MIS is hard, maybe we should restrict the search space?
            
            # Let's check time.
            if time.time() > deadline - 10:
                break
                
            # Efficient bitset creation:
            # Use numpy to create an array of 64-bit integers?
            # Or just use the Python loop with optimized inner logic.
            
            # Actually, for n=5500, the MIS solver itself might be slow if not optimized.
            # Tomita's algorithm with coloring is best.
            
            # Let's implement Tomita's algorithm.
            
            # First, build the bitsets.
            # To speed up bitset creation, we can use the following:
            # For each row, find the set bits.
            
            # Let's try to be smart.
            # If n > 2000, this might be slow.
            # Let's limit w to 3 and 4 first.
            
            # Building bits:
            # We can use:
            # row = comp_mask[i]
            # indices = np.nonzero(row)[0]
            # bits = 0
            # for j in indices: bits |= 1 << j
            # This is O(k) per row, where k is degree.
            
            adj_bits = []
            for i in range(n):
                row = comp_mask[i]
                # Only set bits for j < i? No, graph is undirected.
                # We need full adjacency for the solver.
                indices = np.nonzero(row)[0]
                bits = 0
                # Optimization: if len(indices) is large, this loop is slow.
                # But n is small enough.
                for j in indices:
                    bits |= (1 << j)
                adj_bits.append(bits)
            
            if time.time() > deadline - 5:
                continue
            
            # Tomita's Algorithm for Max Clique
            # Reference: Tomita, Imahori, Hariri (2003)
            
            # We work on the graph G where edges represent compatibility.
            # We want Max Clique.
            
            # Global best
            best_clique_size = 0
            best_clique_nodes = []
            
            # Color the graph to get upper bounds.
            # Greedy coloring.
            
            def greedy_coloring(vertices, adj_bits):
                # vertices: list of node indices
                # Returns: list of (node, color)
                # Colors are 1..k
                # We want to order vertices such that colors are small.
                
                colors = {}
                # Simple greedy:
                # Order vertices by degree descending?
                # For now, just iterate.
                
                # To be efficient, we should process vertices in a specific order.
                # Tomita uses a specific ordering based on coloring.
                
                # Let's implement the standard sequential coloring.
                # We maintain a list of uncolored vertices.
                # Assign smallest possible color.
                
                # This is O(n^2) which is fine.
                
                # Actually, for the bound, we just need the number of colors used.
                # And the vertices grouped by color.
                
                # Let's return the number of colors and the list of vertices sorted by color.
                
                color_class = []
                used_colors = 0
                
                # Sort by degree descending to improve coloring quality?
                # deg = [bin(adj_bits[v]).count('1') for v in vertices]
                # sorted_verts = sorted(vertices, key=lambda v: -deg[v])
                
                # For simplicity, use given order.
                for v in vertices:
                    # Find smallest color not used by neighbors
                    neighbor_colors = set()
                    # Neighbors of v in the current subgraph
                    # We need to know which neighbors are already colored.
                    # This requires maintaining the color of each node.
                    
                    # This standard greedy coloring is a bit complex to implement inline with bitsets efficiently in Python without C extensions.
                    
                    # Alternative: Use a simple bound: size of vertices.
                    # This makes the search exponential.
                    
                    # Let's try a simpler heuristic: 
                    # If n < 100, use brute force / simple branch and bound.
                    # If n > 100, we need a better bound.
                    
                    pass
                return used_colors, vertices
            
            # Given the complexity of implementing a high-performance MIS solver in pure Python within 120s for n=5500,
            # and the fact that w=2 gives 220, let's see if w=3 can beat 220.
            # If w=3 MIS is < 220, we stick with 220.
            
            # Let's run a fast heuristic MIS for w=3 and w=4.
            # Greedy MIS:
            # 1. Sort nodes by degree (in conflict graph) ascending? (Lowest degree first is good for packing).
            #    Degree in conflict graph = number of incompatible nodes.
            #    We want to pick nodes that conflict with few others.
            
            # Conflict degree:
            # conflict_mask = ~comp_mask
            # deg_conflict = np.sum(conflict_mask, axis=1)
            
            if time.time() > deadline - 5:
                continue

            # Heuristic: Greedy Independent Set
            # Sort by conflict degree ascending.
            # Pick node, remove it and all its neighbors (in conflict graph).
            
            # We need the conflict graph bitsets.
            # conflict_bits[i] = ~adj_bits[i] & ((1<<n)-1)
            # But we can just use the comp_mask logic.
            
            # Let's do the greedy on the numpy arrays directly to avoid bitset overhead for the heuristic.
            
            # 1. Compute conflict degrees
            # conflict[i][j] is True if dist^2 < w.
            # conflict[i][j] = dots[i][j] > w/2.0
            conflict_mask = dots > thresh
            np.fill_diagonal(conflict_mask, False)
            
            deg = np.sum(conflict_mask, axis=1)
            
            # Order by degree
            order = np.argsort(deg)
            
            selected = np.zeros(n, dtype=bool)
            available = np.ones(n, dtype=bool)
            
            for i in order:
                if time.time() > deadline - 2:
                    break
                if not available[i]:
                    continue
                # Select i
                selected[i] = True
                # Remove i and all its conflicts from available
                # available = available & ~conflict_mask[i]
                # available[i] = False
                available[conflict_mask[i]] = False
                available[i] = False
                
            k_found = np.sum(selected)
            
            if k_found > best_k:
                best_k = k_found
                best_vecs = vecs[selected]
                
            # Try a second heuristic: Random restarts?
            # Or a simple local search improvement?
            # Given time constraints, let's stick to the greedy.
            
            # For w=2, greedy should find 220 (since all are compatible, deg=0, picks all).
            # For w=3, greedy might find a good subset.
            
            # Let's also try to improve the greedy solution with a simple "fill" or "swap".
            # But time is tight.
            
    if best_vecs is None:
        # Fallback to w=2 all vectors
        vecs = generate_vectors_for_norm_sq(2)
        best_vecs = vecs
        best_k = len(vecs)
        
    return best_vecs

# EVOLVE-BLOCK-END
