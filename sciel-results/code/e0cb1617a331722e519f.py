import itertools
import time
import random
import sys
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # Helper: Greedy construction to get a valid starting point (Fallback)
    def greedy_construct():
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                return selected_blocks
            
            # Pick a random uncovered t-subset
            if uncovered_count < 100:
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = [b_idx for b_idx, b_tsubs in enumerate(block_tsubs) if s_idx in b_tsubs]
            
            best_block_idx = -1
            best_gain = -1
            
            # To speed up, sample candidates if there are too many
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
                
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Helper: Remove redundant blocks
    def reduce_blocks(block_indices):
        block_indices = list(set(block_indices))
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 2.0:
                break
            # Count coverage
            cover_count = [0] * num_tsubs
            for b_idx in block_indices:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_block_indices = []
            for b_idx in block_indices:
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    changed = True
            block_indices = new_block_indices
        return block_indices

    # Attempt ILP Solution
    ilp_solution = None
    try:
        # Build the constraint matrix A where A[i, j] = 1 if block j covers t-subset i
        # We use a sparse representation conceptually, but milp expects arrays.
        # num_tsubs ~ 11628, num_blocks ~ 75582.
        # Matrix size: 11628 x 75582. This is large but sparse.
        # scipy.optimize.milp accepts sparse matrices? 
        # Documentation says: "A: LinearConstraint... A is a 2D array or sparse matrix."
        # Let's try to build a sparse matrix.
        
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        for j, b_tsubs in enumerate(block_tsubs):
            for i in b_tsubs:
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = csr_matrix((data, (rows, cols)), shape=(num_tsubs, num_blocks))
        
        # Objective: minimize sum(x_j)
        c = np.ones(num_blocks)
        
        # Constraints: A @ x >= 1
        # milp requires constraints in the form lb <= A_ub @ x <= ub
        # So we can use LinearConstraint(A, lb=np.ones(num_tsubs), ub=np.inf)
        # But wait, standard form for milp is often min c^T x s.t. A_ub x <= b_ub.
        # LinearConstraint allows lb and ub.
        
        constraints = LinearConstraint(A, lb=np.ones(num_tsubs), ub=np.inf)
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(lb=np.zeros(num_blocks), ub=np.ones(num_blocks))
        
        # Integrality: all integer
        integrality = np.ones(num_blocks)
        
        # Run solver
        # HiGHS is the default solver for milp.
        # It might take a long time. We need to handle timeouts or large problems.
        # If the problem is too large for the solver to handle quickly, it might hang or use too much memory.
        # 11k x 75k sparse matrix is manageable for HiGHS usually, but solving time is the issue.
        
        # To be safe, we can try to run it. If it's too slow, we fall back to greedy.
        # However, milp doesn't have a built-in time limit parameter in the simple API?
        # Actually, `options` dict can pass 'time_limit' to HiGHS.
        
        sol = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 130.0})
        
        if sol.status == 0:  # Optimal
            x = sol.x
            # Extract blocks where x_j > 0.5
            ilp_solution = [j for j in range(num_blocks) if x[j] > 0.5]
            
    except Exception as e:
        # If ILP fails or is too slow/memory heavy, fall back
        pass

    # If ILP succeeded, verify and use it
    if ilp_solution is not None:
        # Verify validity
        covered = bytearray([0] * num_tsubs)
        for b_idx in ilp_solution:
            for j in block_tsubs[b_idx]:
                covered[j] = 1
        
        if all(covered):
            # Try to reduce
            reduced = reduce_blocks(ilp_solution)
            if len(reduced) < len(ilp_solution):
                ilp_solution = reduced
            
            # Return ILP solution
            return [list(all_blocks[i]) for i in ilp_solution]
    else:
        # Fallback to Greedy
        initial_blocks = greedy_construct()
        initial_blocks = reduce_blocks(initial_blocks)
        
        # Simple local search improvement on the greedy result
        # Try to swap blocks to reduce size
        current_blocks = set(initial_blocks)
        cover_count = [0] * num_tsubs
        for b_idx in current_blocks:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1
        
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
        # Time limit for local search
        ls_deadline = deadline - 5.0
        
        while get_elapsed() < ls_deadline:
            # Try to remove a redundant block
            removed = False
            for b_idx in list(current_blocks):
                # Check if any t-sub is uniquely covered by b_idx
                is_redundant = True
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    current_blocks.remove(b_idx)
                    for j in block_tsubs[b_idx]:
                        cover_count[j] -= 1
                    removed = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            if not removed:
                # Try a swap: remove b_out, add b_in
                # Pick a random block to remove
                if not current_blocks:
                    break
                b_out = random.choice(list(current_blocks))
                
                # Find critical tsubs
                criticals = [j for j in block_tsubs[b_out] if cover_count[j] == 1]
                
                if not criticals:
                    # Redundant, should have been caught
                    continue
                
                # Find b_in that covers all criticals
                # Candidates: blocks covering criticals[0]
                # We need to precompute tsub_blocks for this
                # This is expensive to do on the fly. 
                # Let's just break if no simple removals found, as building tsub_blocks 
                # is O(num_blocks * 56) which is ~4M ops, acceptable once.
                
                # Actually, let's just do a few random restarts of greedy if time permits
                break

        # If time remains, do another greedy restart
        if get_elapsed() < deadline - 10.0:
            initial_blocks_2 = greedy_construct()
            initial_blocks_2 = reduce_blocks(initial_blocks_2)
            if len(initial_blocks_2) < len(best_blocks):
                best_blocks = initial_blocks_2

        return [list(all_blocks[i]) for i in best_blocks]

