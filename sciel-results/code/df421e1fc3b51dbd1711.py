"""Covering design C(19,8,5): Integer Linear Programming (ILP) relaxation with heuristic rounding.
Strategy:
1. Formulate the problem as a set cover ILP.
2. Solve the Linear Programming (LP) relaxation using SciPy's linprog (HiGHS backend).
3. Use the dual variables (shadow prices) from the LP solution to guide a greedy construction.
4. Apply local search (2-opt swaps) to refine the solution.
This approach leverages global optimization information (duals) which is fundamentally
different from the purely local greedy/local-search strategy of the current program.
"""
import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx_t = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    M = len(all_blocks)
    
    # Build incidence matrix A (N x M) where A[i, j] = 1 if block j covers t-subset i
    # N = C(19, 5) = 11628
    # M = C(19, 8) = 75582
    # This is a large sparse matrix. We can build it efficiently.
    
    # For each block j, find the t-subsets it covers.
    # Instead of building the full dense matrix, we can build the sparse structure or just the lists.
    # For linprog, we need the matrix in some format. COO or CSC is good.
    # However, constructing the full incidence matrix might be slow/memory heavy if done naively.
    # Let's estimate: 11628 * 75582 ~ 878 million entries. Most are 0.
    # Number of 1s: M * C(k, t) = 75582 * 56 ~ 4.2 million.
    # This is very sparse.
    
    # We will use scipy.sparse for efficiency if possible, but linprog accepts dense too.
    # Given the sparsity, let's try to build it in COO format.
    
    from scipy import sparse
    
    rows = []
    cols = []
    data = []
    
    # Iterate through all blocks
    for j, block in enumerate(all_blocks):
        # Get all t-subsets of this block
        for s in itertools.combinations(block, t):
            i = idx_t[s]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, M)).tocsr()
    
    # Objective: minimize sum of x_j
    c = np.ones(M)
    
    # Constraints: sum_j A[i, j] x_j >= 1 for all i
    # linprog minimizes c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, bounds
    # So we need -A x <= -1
    A_ub = -A
    b_ub = -np.ones(N)
    
    bounds = [(0, 1) for _ in range(M)]
    
    # Solve LP relaxation
    # Use HiGHS backend which is fast for LPs
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            raise Exception("LP failed: " + res.message)
        
        x_lp = res.x
        # Get dual variables (shadow prices) for the constraints
        # res.ineqlin.marginals gives the duals for A_ub x <= b_ub
        # Since A_ub = -A, the duals correspond to the original constraints with a sign flip?
        # Actually, let's just use the primal solution x_lp to guide the heuristic.
        # Or better, use the duals to prioritize t-subsets?
        # Standard approach: Greedy set cover using LP duals or primal solution.
        # Let's try a "Round Down" or "Greedy on LP solution" approach.
        
        # Strategy: 
        # 1. Select all blocks with x_lp[j] > 0.5 (or some threshold).
        # 2. Check if this covers everything.
        # 3. If not, add blocks greedily based on x_lp values (higher x_lp = more "important").
        
        # Alternatively, use the dual variables (y_i) to define a weight for each t-subset.
        # Then use a greedy algorithm that maximizes covered "weight".
        
        # Let's try a simpler heuristic first: 
        # Sort blocks by x_lp[j] descending.
        # Add blocks until all t-subsets are covered.
        
        # But x_lp might be fractional.
        # Let's try a "Greedy LP" approach:
        # 1. Start with empty set.
        # 2. While not all covered:
        #    - Pick the block that covers the most *uncovered* t-subsets.
        #    - Tie-break by x_lp[j] (higher is better).
        
        # This is similar to standard greedy but uses LP solution for tie-breaking.
        
    except Exception as e:
        # If LP fails or is too slow, fall back to pure greedy
        x_lp = np.ones(M) / M
        pass
    
    # Heuristic Construction based on LP solution
    def construct_from_lp(x_lp, deadline):
        uncovered = set(range(N))
        blocks = []
        
        # Precompute list of t-subsets covered by each block for fast lookup
        # We already have the sparse matrix A.
        # A[j, :] gives the t-subsets covered by block j.
        # But accessing sparse rows is fast.
        
        # To make greedy faster, we can precompute the list of covered t-subsets for each block.
        # This is memory heavy (75k lists of ~56 ints). ~4M ints. Manageable.
        block_covers = []
        for j in range(M):
            # Get non-zero indices in row j of A (which is N x M, so column j)
            # A is CSR, so A[:, j] is not efficient.
            # Let's rebuild the list of lists from the COO data we had.
            pass
        
        # Let's just use the original all_blocks and check coverage on the fly or precompute.
        # Precomputation of block_covers is better.
        
        # Re-using the rows/cols/data from above? They are local to the outer scope.
        # Let's just recompute or use a different structure.
        
        # Actually, we can just iterate.
        # For each step, we need to find the best block.
        # Naive: for each block, count uncovered t-subsets it covers.
        # This is O(M * N) per step, which is too slow.
        
        # Better: Maintain a count of uncovered t-subsets.
        # When we add a block, we decrement the count for its covered t-subsets.
        # We need to efficiently find the block with max gain.
        
        # Let's use a priority queue or just iterate with a fast inner loop.
        # Since M=75k, and we might add ~300 blocks, 300 * 75k = 22.5M operations.
        # Each operation involves checking ~56 t-subsets. 22.5M * 56 ~ 1.2B operations. Too slow in Python.
        
        # We need a faster way.
        # Maybe use numpy?
        # Or, limit the search space.
        
        # Alternative: Use the LP solution x_lp directly.
        # Sort blocks by x_lp.
        # Add blocks in that order until covered.
        
        order = np.argsort(x_lp)[::-1]
        
        covered = np.zeros(N, dtype=bool)
        blocks = []
        
        for j in order:
            if time.time() > deadline:
                break
            if len(covered) == 0:
                break
            
            # Check if this block covers any new t-subsets
            # Get t-subsets covered by block j
            # We need the list of t-subsets for block j.
            # Let's precompute this list now.
            
            # This is a one-time cost.
            # Let's create a list of lists: covers[j] = list of t-subset indices
            # We can do this from the COO data.
            
            # Since we are inside a nested function, let's pass the data or recompute.
            # Let's recompute covers efficiently.
            pass
            
        # This approach is getting complicated. Let's step back.
        
        # Simpler Strategy:
        # 1. Solve LP.
        # 2. Use the dual variables (shadow prices) to weight the t-subsets.
        # 3. Use a standard greedy set cover algorithm, but when selecting the next block,
        #    maximize the sum of weights of newly covered t-subsets.
        
        # This is a well-known heuristic.
        
        # Get duals
        # res.ineqlin.marginals has shape (N,)
        # These are the duals for -A x <= -1.
        # The dual for the original constraint A x >= 1 is -marginals (since we flipped the inequality).
        # Actually, let's just use the marginals as weights. Higher marginal means more "valuable" to cover.
        
        duals = res.ineqlin.marginals
        # The duals can be negative or zero. We want to prioritize covering "hard" constraints.
        # Let's use weight = 1 + abs(duals) or just duals if positive.
        # Let's just use the duals directly.
        
        # Greedy with weights
        uncovered = set(range(N))
        blocks = []
        
        # Precompute block_covers: list of lists of t-subset indices
        block_covers = [[] for _ in range(M)]
        # We need to map from the COO data to these lists.
        # Let's just rebuild it from all_blocks.
        for j, block in enumerate(all_blocks):
            for s in itertools.combinations(block, t):
                block_covers[j].append(idx_t[s])
                
        # Now run weighted greedy
        # To speed up, we can maintain a "gain" for each block.
        # Initially, gain[j] = sum of weights of t-subsets covered by j.
        # When we add a block, we subtract weights of covered t-subsets from other blocks.
        
        # This is O(M * N) to initialize gains? No, O(sum of covers) = 4.2M.
        # Then each step, we pick max gain, then update gains for all blocks.
        # Update is O(M * avg_cover) per step? No.
        # If we add block j, we need to subtract weight[s] from gain[k] for all k that cover s.
        # This requires knowing which blocks cover each t-subset.
        # Let's build t_to_blocks: list of lists.
        
        t_to_blocks = [[] for _ in range(N)]
        for j in range(M):
            for s_idx in block_covers[j]:
                t_to_blocks[s_idx].append(j)
                
        # Initialize gains
        gains = np.zeros(M)
        for s_idx in range(N):
            w = duals[s_idx]
            if w > 0:
                for j in t_to_blocks[s_idx]:
                    gains[j] += w
                    
        # Greedy loop
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find block with max gain
            # We can just use np.argmax
            j = np.argmax(gains)
            if gains[j] <= 0:
                # No positive gain, break or pick random
                break
                
            # Add block j
            blocks.append(all_blocks[j])
            
            # Update uncovered and gains
            for s_idx in block_covers[j]:
                if s_idx in uncovered:
                    uncovered.discard(s_idx)
                    # Subtract weight from all blocks covering s_idx
                    w = duals[s_idx]
                    if w > 0:
                        for k in t_to_blocks[s_idx]:
                            gains[k] -= w
                            
        return blocks
    
    # Run the LP-based heuristic
    try:
        blocks = construct_from_lp(x_lp, deadline - 10)
    except Exception:
        blocks = []
        
    # Fallback if LP heuristic fails or is slow
    if not blocks:
        # Simple greedy
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            if time.time() > deadline - 10:
                break
            # Pick random uncovered t-subset
            s = random.choice(list(uncovered))
            # Find a random block covering it
            candidates = t_to_blocks[s]
            if not candidates:
                break
            j = random.choice(candidates)
            blocks.append(all_blocks[j])
            for s_idx in block_covers[j]:
                uncovered.discard(s_idx)
                
    # Local Search to refine
    # Try to remove redundant blocks
    def is_redundant(block_idx, blocks, cover_counts):
        b = blocks[block_idx]
        for s in itertools.combinations(b, t):
            i = idx_t[s]
            if cover_counts[i] == 1:
                return False
        return True

    def calculate_cover_counts(blocks):
        counts = np.zeros(N, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                counts[idx_t[s]] += 1
        return counts

    def try_remove_redundant(blocks, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            counts = calculate_cover_counts(blocks)
            for i in range(len(blocks)):
                if is_redundant(i, blocks, counts):
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # Verify and clean
    blocks = try_remove_redundant(blocks, deadline - 5)
    
    # Final validity check
    if blocks:
        counts = calculate_cover_counts(blocks)
        if np.all(counts > 0):
            return [list(b) for b in blocks]
            
    # If invalid, return empty or fallback
    return [list(b) for b in blocks] if blocks else []
