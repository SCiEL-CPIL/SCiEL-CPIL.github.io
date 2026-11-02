"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pools
    # Mix of weight-2 and weight-4 vectors.
    # Weight 2: norm^2 = 2. Count = C(11,2)*4 = 220.
    # Weight 4: norm^2 = 4. Count = C(11,4)*16 = 5280.
    # Total N = 5500.
    
    candidates = []
    types = [] # 0 for w2, 1 for w4
    
    # Generate Weight 2 vectors
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[comb[0]] = signs[0]
            v[comb[1]] = signs[1]
            candidates.append(v)
            types.append(0)
            
    # Generate Weight 4 vectors
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            types.append(1)
            
    candidates = np.array(candidates)
    types = np.array(types)
    n_cand = len(candidates)
    
    # 2. Build Conflict Graph
    # Condition: ||v_i - v_j||^2 >= 4
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
    # Let n_i = ||v_i||^2. Conflict if n_i + n_j - 2 v_i.v_j < 4.
    
    # Compute Gram matrix G = V V^T
    G = candidates @ candidates.T
    
    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Distance squared matrix D_sq = n_i + n_j - 2 G
    # We need D_sq >= 4 for compatibility.
    # Conflict if D_sq < 4.
    
    # To save memory and speed up bitmask conversion, we can compute the conflict matrix directly.
    # D_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*G[i, j]
    
    # Broadcasting: (N, 1) + (1, N) - 2 * (N, N)
    # This creates a (5500, 5500) float array. 5500^2 * 8 bytes ~ 240 MB. Acceptable.
    # However, we can be smarter.
    
    # Conflict if:
    # w2-w2 (2,2): 4 - 2G < 4 => G > 0
    # w2-w4 (2,4): 6 - 2G < 4 => 2G > 2 => G > 1
    # w4-w4 (4,4): 8 - 2G < 4 => 2G > 4 => G > 2
    
    # Create type matrix
    # types is (N,). type_mat[i,j] = types[i] + types[j]
    # 0: 0+0 (w2-w2)
    # 1: 0+1 or 1+0 (w2-w4)
    # 2: 1+1 (w4-w4)
    
    # We can just compute the condition directly using broadcasting on types and G.
    # But constructing the full (N,N) boolean matrix is the bottleneck for memory if not careful, 
    # but 5500x5500 bool is ~30MB, which is fine.
    
    # Let's compute the conflict mask.
    # We iterate over type pairs to avoid large temporary arrays if possible, 
    # but numpy vectorization is fast.
    
    # Method:
    # Create a matrix of thresholds.
    # thresh[i,j] = 0 if type_i==0 and type_j==0
    #              = 1 if (type_i==0 and type_j==1) or (type_i==1 and type_j==0)
    #              = 2 if type_i==1 and type_j==1
    
    # Actually, simpler:
    # conflict = (norms_sq[:, None] + norms_sq[None, :] - 2 * G) < 4
    
    # Let's do this. It's clean and fast in numpy.
    D_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * G
    conflict_mat = D_sq < 4.0
    
    # Ensure diagonal is False (no self-conflict)
    np.fill_diagonal(conflict_mat, False)
    
    # 3. Convert to Bitmasks for Branch and Bound
    # N = 5500. We need 5500 bitmasks of length 5500.
    # 5500 bits = 87 uint64 integers.
    # Total memory: 5500 * 87 * 8 bytes ~ 3.8 MB. Very efficient.
    
    # We will represent the neighborhood of each node i as a list of uint64 masks.
    # adj_masks[i] is a list of 87 uint64s.
    
    num_words = (n_cand + 63) // 64
    adj_masks = [None] * n_cand
    
    # Convert boolean matrix to bitmasks.
    # We can do this row by row.
    for i in range(n_cand):
        row = conflict_mat[i]
        # Convert boolean array to uint64 words
        # Reshape to (-1, 64), pad if necessary
        if n_cand % 64 != 0:
            padded_len = num_words * 64
            row_padded = np.zeros(padded_len, dtype=bool)
            row_padded[:n_cand] = row
        else:
            row_padded = row
            
        # View as uint64? No, bool is 1 byte. 
        # We need to pack 8 bools into 1 byte, then 8 bytes into uint64?
        # Easier: Use numpy packbits? 
        # np.packbits(row_padded) gives bytes (8 bits per byte).
        # Then view as uint64.
        
        packed = np.packbits(row_padded)
        # Ensure length is multiple of 8 for uint64 view
        if len(packed) % 8 != 0:
            # num_words*64 is multiple of 8, so packbits length is num_words*8.
            # This should be fine.
            pass
        
        # View as uint64
        # Endianness matters. Little-endian is standard for numpy on x86.
        # np.packbits produces bytes in order. 
        # If we view as uint64, the first 8 bytes become the first uint64.
        # Bit 0 of the first byte corresponds to index 0.
        # Bit 0 of the uint64 is the least significant bit.
        # So bit 0 of uint64[0] is index 0.
        # This matches standard bit indexing.
        
        masks = np.frombuffer(packed, dtype=np.uint64).copy()
        adj_masks[i] = masks
        
    # 4. Branch and Bound for Maximum Independent Set
    # We want to find a set of nodes such that no two are connected (conflict).
    # This is the Maximum Independent Set problem.
    
    # Heuristic: Maximum Clique in the complement graph is MIS in original.
    # But we can also do MIS directly.
    # Standard B&B for MIS:
    # State: (P, I) where P is set of candidates, I is set of selected.
    # We want to maximize |I|.
    
    # To speed up, we use a greedy coloring bound on the candidate set P.
    # The size of the maximum independent set in P is at most the number of colors needed to color the graph induced by P.
    
    # Since N=5500 is large, a full B&B might be slow if the bound is weak.
    # However, the graph is dense (conflicts are common).
    # Let's check density.
    # w2-w2: dot > 0. 
    # w4-w4: dot > 2.
    # The graph is likely dense, meaning the independence number is small relative to N.
    # Wait, if the graph is dense, the complement is sparse.
    # MIS in dense graph = Max Clique in sparse complement.
    # Max Clique on sparse graphs is often easier to bound?
    
    # Actually, let's look at the "kissing number" context.
    # Kissing number in 11D is 556 (or similar, high).
    # Our pool is 5500.
    # If the answer is ~500, and N=5500, the independence number is ~10% of N.
    # The graph is not extremely dense in terms of non-edges.
    
    # Let's implement a standard Bron-Kerbosch variant or a simple B&B with coloring.
    # Given the time limit, a robust B&B with a good initial solution from greedy is key.
    
    # First, get a good initial solution using Greedy Coloring/Selection.
    # We will use a greedy algorithm that picks the node with the highest degree in the remaining graph?
    # Or lowest degree?
    # For MIS, picking the node with the lowest degree in the induced subgraph is a common heuristic.
    
    # Let's run a fast greedy to get a baseline.
    # We'll use the same "least conflicts" greedy as before but faster using bitmasks.
    
    def greedy_mis():
        remaining = set(range(n_cand))
        selected = []
        # To speed up, we can just iterate through nodes and check conflicts.
        # But N=5500, O(N^2) is 30M ops. In Python, this is slow (~3-5s).
        # We can do better with bitmasks.
        
        # Let's just use the previous numpy-based greedy which was fast enough for 100 restarts.
        # It took < 1s per restart. 100 restarts = 100s. Too slow for B&B setup.
        # We need 1 good initial solution.
        
        # Let's do 10 fast random restarts in numpy to get a good lower bound.
        best_sel = []
        for _ in range(10):
            perm = np.random.permutation(n_cand)
            curr_sel = []
            for idx in perm:
                if not curr_sel:
                    curr_sel.append(idx)
                else:
                    if not np.any(conflict_mat[idx, curr_sel]):
                        curr_sel.append(idx)
            if len(curr_sel) > len(best_sel):
                best_sel = curr_sel
        return best_sel

    initial_solution = greedy_mis()
    best_size = len(initial_solution)
    
    # Now, B&B.
    # We will use a recursive search.
    # To avoid recursion depth issues and overhead, we might need to be careful.
    # But with good bounds, depth should be limited.
    
    # We need a function to compute the coloring bound for a set of nodes.
    # This is expensive if done naively.
    # Standard trick: Use a greedy coloring on the candidate set.
    # The number of colors is an upper bound on the size of the MIS in that set.
    
    # Let's implement a specialized B&B.
    # State:
    #   candidates: list of node indices
    #   selected: list of node indices (implicit in recursion depth)
    
    # Optimization:
    # We can pre-sort nodes by degree to help pruning?
    
    # Given the complexity of implementing a high-performance B&B in Python within 150s for N=5500,
    # and the risk of it being slower than a well-tuned heuristic,
    # let's try a **Local Search with Bitmasks** approach which is structurally different from the simple greedy restarts.
    
    # Actually, the prompt asks for a MATERIAL change.
    # B&B is the right direction. Let's try to make it work.
    
    # If B&B is too slow, we return the greedy result.
    
    # Let's define the B&B function.
    # We'll use a global variable for the best found.
    
    global_best = initial_solution.copy()
    global_best_size = best_size
    
    def bit_count(mask_list):
        return sum(m.bit_count() for m in mask_list)
        
    def intersect_masks(mask1, mask2):
        return [m1 & m2 for m1, m2 in zip(mask1, mask2)]
        
    def union_masks(mask1, mask2):
        return [m1 | m2 for m1, m2 in zip(mask1, mask2)]
        
    def diff_masks(mask1, mask2):
        return [m1 & ~m2 for m1, m2 in zip(mask1, mask2)]
        
    # Greedy Coloring Bound
    # Input: list of candidate node indices
    # Output: upper bound on MIS size in this set
    def coloring_bound(candidates):
        if not candidates:
            return 0
            
        # Greedy coloring
        # We need to color the graph induced by candidates.
        # This is O(|C|^2) in worst case.
        # If |C| is large, this is slow.
        # We only call this if |C| is manageable?
        # Or we use a fast approximation.
        
        # Let's use a simple heuristic bound: 
        # Sum of (1 / (1 + degree_in_induced_subgraph)) is a lower bound on independence number?
        # No, we need an UPPER bound.
        # A simple upper bound is the maximum degree + 1? No.
        # Turan's theorem?
        
        # Let's stick to the standard: Greedy Coloring.
        # To make it fast, we limit the candidates considered for coloring?
        # No, the bound must be valid for the whole set.
        
        # Given the time constraints and Python overhead, a full B&B with coloring bound 
        # on 5500 nodes is likely to time out unless the tree is very small.
        
        # Alternative: **Simulated Annealing** on the bitmask representation.
        # This is a different optimization method (stochastic global search vs deterministic greedy/restart).
        
        # Let's switch to Simulated Annealing.
        # State: Bitmask of selected nodes.
        # Move: Add a node, Remove a node, Swap a node.
        # Constraint: Must be an independent set.
        # We can maintain the set and check validity of moves efficiently.
        
        # Let's implement Simulated Annealing.
        
        rng = np.random.default_rng(42)
        
        # Start with the best greedy solution
        current_set = set(initial_solution)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        
        # Precompute conflict masks for fast checking?
        # We already have adj_masks.
        
        # To check if a node u can be added to S:
        # u must not conflict with any v in S.
        # i.e., for all v in S, bit u in adj_masks[v] is 0.
        # Or, (adj_masks[u] & mask_S) == 0.
        
        # We need to maintain mask_S (list of uint64).
        
        def set_to_mask(node_set):
            mask = [0] * num_words
            for node in node_set:
                word_idx = node // 64
                bit_idx = node % 64
                mask[word_idx] |= (1 << bit_idx)
            return mask
            
        def mask_to_set(mask):
            nodes = []
            for w_idx, m in enumerate(mask):
                if m:
                    # Extract bits
                    # Use bit_length or iteration
                    x = m
                    while x:
                        # Get lowest set bit
                        lsb = x & -x
                        bit_pos = lsb.bit_length() - 1
                        nodes.append(w_idx * 64 + bit_pos)
                        x ^= lsb
            return nodes
            
        current_mask = set_to_mask(current_set)
        
        # Helper: Check if node u is compatible with current_mask
        def is_compatible(u, mask):
            adj_u = adj_masks[u]
            # Check intersection
            for w in range(num_words):
                if adj_u[w] & mask[w]:
                    return False
            return True
            
        # Helper: Add node u to mask
        def add_to_mask(mask, u):
            new_mask = mask.copy()
            word_idx = u // 64
            bit_idx = u % 64
            new_mask[word_idx] |= (1 << bit_idx)
            return new_mask
            
        # Helper: Remove node u from mask
        def remove_from_mask(mask, u):
            new_mask = mask.copy()
            word_idx = u // 64
            bit_idx = u % 64
            new_mask[word_idx] &= ~(1 << bit_idx)
            return new_mask
            
        # Simulated Annealing Parameters
        T = 100.0
        T_min = 1.0
        cooling_rate = 0.99
        max_iter = 500000 # Adjust based on time
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # Choose a move: Add, Remove, or Swap
            move_type = rng.integers(0, 3)
            
            if move_type == 0: # Add
                # Pick random node not in current_set
                # To do this efficiently, we can pick from a pool of unselected nodes.
                # Since N=5500, picking random and checking if in set is O(1) if we use a boolean array or set.
                # We have current_set.
                u = rng.integers(0, n_cand)
                if u in current_set:
                    continue
                    
                # Check if u is compatible
                if is_compatible(u, current_mask):
                    new_set = current_set | {u}
                    new_mask = add_to_mask(current_mask, u)
                    new_size = current_size + 1
                    
                    # Always accept improvement
                    current_set = new_set
                    current_mask = new_mask
                    current_size = new_size
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
                        
            elif move_type == 1: # Remove
                # Pick random node from current_set
                if not current_set:
                    continue
                u = rng.choice(list(current_set))
                
                new_set = current_set - {u}
                new_mask = remove_from_mask(current_mask, u)
                new_size = current_size - 1
                
                # Accept with probability exp(-delta/T) where delta = new_size - current_size = -1
                # p = exp(1/T) > 1, so always accept? 
                # No, we want to escape local minima. 
                # Standard SA accepts worse moves with prob exp(-delta/T).
                # Here delta = -1 (worse). p = exp(1/T). Since T>0, p>1.
                # This means we ALWAYS accept removals? That would just empty the set.
                # We need to be careful with the objective.
                # Objective: Maximize size.
                # Delta = new_size - old_size.
                # If delta < 0, accept with prob exp(delta/T).
                # exp(-1/T).
                
                delta = new_size - current_size
                if delta >= 0:
                    current_set = new_set
                    current_mask = new_mask
                    current_size = new_size
                else:
                    if rng.random() < np.exp(delta / T):
                        current_set = new_set
                        current_mask = new_mask
                        current_size = new_size
                        
            else: # Swap
                # Pick u in current_set, v not in current_set
                if not current_set:
                    continue
                u = rng.choice(list(current_set))
                v = rng.integers(0, n_cand)
                if v in current_set:
                    continue
                
                # Check if v is compatible with (current_set - {u})
                # i.e., v is not in current_set AND v is compatible with all w in current_set \ {u}
                # This is equivalent to:
                # 1. v is not in current_set
                # 2. (adj_masks[v] & (current_mask without u)) == 0
                
                # Let's check compatibility with current_set first.
                # If v is compatible with current_set, we could just add it (handled in Add).
                # But we are swapping.
                # We need to check if v conflicts with any w in current_set \ {u}.
                
                # Efficient check:
                # Let M_u = mask with only u set.
                # Let M_current = current_mask.
                # M_rest = M_current & ~M_u.
                
                # Check if (adj_masks[v] & M_rest) == 0.
                
                # Also, we don't need to check if v conflicts with u, because u is removed.
                
                # Construct M_u
                word_idx_u = u // 64
                bit_idx_u = u % 64
                M_u = [0] * num_words
                M_u[word_idx_u] = (1 << bit_idx_u)
                
                M_rest = [m & ~m_u for m, m_u in zip(current_mask, M_u)]
                
                # Check compatibility of v with M_rest
                compatible = True
                adj_v = adj_masks[v]
                for w in range(num_words):
                    if adj_v[w] & M_rest[w]:
                        compatible = False
                        break
                        
                if compatible:
                    new_set = (current_set - {u}) | {v}
                    new_mask = remove_from_mask(current_mask, u)
                    new_mask = add_to_mask(new_mask, v)
                    new_size = current_size # Size unchanged
                    
                    # Size is same. We accept to explore.
                    # To avoid oscillation, we can require some other metric or just accept with prob 0.5?
                    # Or accept if it leads to a "better" structure?
                    # For now, accept with prob 0.5 to allow movement.
                    if rng.random() < 0.5:
                        current_set = new_set
                        current_mask = new_mask
                        # current_size unchanged
                        
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # Return best found
        if best_set:
            # Convert best_set to array
            # Ensure it's a valid independent set (sanity check)
            best_list = list(best_set)
            final_vecs = candidates[best_list]
            return final_vecs
        else:
            return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)

# EVOLVE-BLOCK-END
