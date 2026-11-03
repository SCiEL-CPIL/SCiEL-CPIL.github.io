import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible blocks
    # A block is a k-subset of {0,...,v-1}
    # For each block, compute which t-subsets it covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the set of t-subset indices it covers
    # Using a list of sets for fast intersection checks
    block_to_covered = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
        block_to_covered.append(covered)
    
    # Build reverse index: for each t-subset, which blocks cover it?
    # This is essential for the Pricing Subproblem and Greedy
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)
    
    def greedy_cover(seed=0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            
            # Pick a random uncovered t-subset to guide the search
            start_idx = rng.choice(list(uncovered))
            
            # Find all blocks that cover this t-subset
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(all_blocks[best_block])
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        
        return blocks

    def pricing_subproblem(uncovered_set):
        """
        Find the block that covers the most elements in uncovered_set.
        Returns (best_block_idx, best_gain).
        If uncovered_set is empty, returns (None, 0).
        """
        if not uncovered_set:
            return None, 0
        
        # Heuristic: Start with a random element from uncovered_set
        # to limit the candidate pool, or check all if set is small?
        # Given n_t = 2024, and each block covers 35, the intersection is small.
        # But checking all 346k blocks is too slow for many iterations.
        # Strategy: Sample a few random uncovered triples and find the best block for each.
        # Then take the best among those.
        
        best_block_idx = None
        best_gain = 0
        
        # Sample up to 5 random uncovered triples to keep pricing fast
        sample_size = min(5, len(uncovered_set))
        sampled_indices = random.sample(list(uncovered_set), sample_size)
        
        candidates = set()
        for idx in sampled_indices:
            candidates.update(t_subset_to_blocks[idx])
        
        for b_idx in candidates:
            gain = len(block_to_covered[b_idx] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        return best_block_idx, best_gain

    def solve_column_generation(initial_blocks):
        """
        Iteratively add columns (blocks) to the ILP and solve.
        """
        try:
            import pulp
        except ImportError:
            return initial_blocks

        # Start with the initial set of blocks from greedy
        active_blocks = set()
        for b in initial_blocks:
            # Find index of this block
            # Since all_blocks is a list of tuples, we can't do O(1) lookup easily without a dict
            # But we can just use the block tuple itself as the key in the ILP variables later
            # For now, let's just keep the block tuples.
            pass
        
        # We need a mapping from block tuple to index for the precomputed structures
        # But for ILP, we can just use the block tuple as the variable name or a generated ID.
        # Let's maintain a list of active block indices.
        
        # Map block tuple to index for quick lookup
        # This is expensive to build fully, but we only need it for blocks we add.
        # Actually, we already have all_blocks. Let's just track indices.
        
        # Initialize active set with indices from greedy
        active_indices = []
        # We need to map the block tuples from greedy back to indices.
        # Since greedy returns all_blocks[i], we can just store indices in greedy.
        # Let's modify greedy to return indices? No, let's keep it simple.
        # We'll just re-derive the indices.
        
        # Better: Let greedy return indices.
        # Let's refactor greedy to return indices.
        
        # Re-implementing greedy to return indices for efficiency in this function
        def greedy_indices(seed=0):
            rng = random.Random(seed)
            uncovered = set(range(n_t))
            indices = []
            
            while uncovered:
                if time.time() > deadline - 20.0:
                    break
                
                start_idx = rng.choice(list(uncovered))
                candidate_blocks = t_subset_to_blocks[start_idx]
                
                best_block = None
                best_gain = -1
                for b_idx in candidate_blocks:
                    gain = len(block_to_covered[b_idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                
                if best_block is None:
                    break
                    
                indices.append(best_block)
                for ts_idx in block_to_covered[best_block]:
                    uncovered.discard(ts_idx)
            
            return indices

        active_indices = greedy_indices(0)
        
        # Ensure uniqueness
        active_indices = list(set(active_indices))
        
        current_solution_blocks = [all_blocks[i] for i in active_indices]
        
        max_iter = 20
        for _ in range(max_iter):
            if time.time() > deadline - 15.0:
                break
                
            # 1. Solve ILP with current active blocks
            prob = pulp.LpProblem("CoveringDesign", pulp.LpMinimize)
            
            x = {}
            for idx in active_indices:
                x[idx] = pulp.LpVariable(f"x_{idx}", cat=pulp.LpBinary)
            
            prob += pulp.lpSum(x[idx] for idx in active_indices)
            
            # Constraints: each t-subset must be covered by at least one active block
            # If a t-subset is not covered by any active block, the ILP will be infeasible.
            # We must ensure the initial greedy solution covers everything.
            # The pricing step adds blocks, so we should check coverage before solving?
            # Or just let ILP fail and add more?
            # Better: Check coverage of current active blocks. If not full, add blocks via pricing until full.
            
            # Check coverage of current active blocks
            covered_by_active = set()
            for idx in active_indices:
                covered_by_active.update(block_to_covered[idx])
            
            if len(covered_by_active) < n_t:
                # Need to add blocks to make it feasible
                uncovered = set(range(n_t)) - covered_by_active
                # Add blocks via pricing until covered
                for _ in range(100): # Safety loop
                    if not uncovered:
                        break
                    if time.time() > deadline - 15.0:
                        break
                    b_idx, _ = pricing_subproblem(uncovered)
                    if b_idx is None:
                        break
                    if b_idx not in active_indices:
                        active_indices.append(b_idx)
                    # Update coverage
                    covered_by_active.update(block_to_covered[b_idx])
                    uncovered -= block_to_covered[b_idx]
                
                if uncovered:
                    # Still not covered? Fallback to full greedy
                    return greedy_cover(0)
                
                # Continue to next iteration to solve ILP on this larger set
                continue
            
            # Now all t-subsets are covered by active blocks. Solve ILP.
            for i in range(n_t):
                covering_blocks = [idx for idx in active_indices if i in block_to_covered[idx]]
                if covering_blocks:
                    prob += pulp.lpSum(x[idx] for idx in covering_blocks) >= 1
                else:
                    # This shouldn't happen if coverage check passed
                    continue
            
            solver_time = min(10.0, deadline - time.time() - 10.0)
            if solver_time <= 0:
                break
                
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time)
            prob.solve(solver)
            
            solution_indices = []
            for idx in active_indices:
                if x[idx].value() is not None and x[idx].value() > 0.5:
                    solution_indices.append(idx)
            
            if not solution_indices:
                break
                
            # 2. Pricing Step: Find a block NOT in active_indices that has a negative reduced cost.
            # Reduced cost for a covering problem:
            # Dual variables y_i >= 0 for each constraint (t-subset i).
            # Cost of block j = 1 - sum(y_i for i in block j).
            # We want to find a block j such that 1 - sum(y_i) < 0, i.e., sum(y_i) > 1.
            
            # Get dual values
            duals = {}
            for i in range(n_t):
                # The constraint in pulp is usually named or accessible via prob.constraints
                # This is tricky in PuLP. Let's use a simpler heuristic pricing if duals are hard to get.
                # Heuristic: Find the block that covers the most "hard" triples.
                # Which triples are "hard"? Those covered by few blocks in the current solution.
                pass
            
            # Alternative Pricing: 
            # Let's just find the block that maximizes the number of uncovered triples 
            # if we were to remove the least useful blocks? No.
            
            # Let's stick to the dual if possible.
            # In PuLP, constraints are in prob.constraints.
            # The order might be tricky.
            
            # Simpler robust pricing:
            # Calculate for each candidate block (not in active) the "value" based on current solution.
            # Actually, the standard column generation for set cover uses duals.
            # If we can't easily get duals, we can use a heuristic:
            # Identify triples that are covered by exactly 1 block in the current ILP solution.
            # These are "critical". Find a block that covers many critical triples.
            
            critical_triples = []
            for i in range(n_t):
                count = 0
                for idx in solution_indices:
                    if i in block_to_covered[idx]:
                        count += 1
                if count == 1:
                    critical_triples.append(i)
            
            if not critical_triples:
                # No critical triples, solution might be robust. Stop or try random pricing.
                # If no critical triples, the current solution is likely optimal or near-optimal for this set.
                # Try to find any block that could replace a block?
                # Let's just stop if no critical triples.
                break
            
            # Find a block (not in active) that covers the most critical triples
            # We can't check all 346k blocks.
            # Use the same sampling strategy: pick a random critical triple, find candidates.
            
            best_new_block = None
            best_gain = 0
            
            # Sample critical triples
            sample_size = min(10, len(critical_triples))
            sampled_crit = random.sample(critical_triples, sample_size)
            
            candidates = set()
            for idx in sampled_crit:
                candidates.update(t_subset_to_blocks[idx])
            
            # Filter out active blocks
            candidates -= set(active_indices)
            
            for b_idx in candidates:
                # Count how many critical triples it covers
                gain = len(block_to_covered[b_idx] & set(critical_triples))
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = b_idx
            
            if best_new_block is None:
                # No improvement found via critical triples
                # Try random pricing: pick a random block not in active
                # This is expensive. Let's just break.
                break
            
            # Add the new block
            active_indices.append(best_new_block)
            
            # Update current solution blocks for next iteration
            # The ILP solution might change next time.
            
        # Final solution from the last ILP run
        # We need to re-solve or extract from the last prob?
        # The loop overwrites prob. We should store the best solution found.
        
        # Let's restructure to store best solution.
        # For now, return the blocks from the last active set that were selected by ILP.
        # If the loop broke early, we might not have a valid ILP solution from the last state.
        # Safest: Return the best feasible solution found.
        
        # Let's just return the blocks from the greedy solution if ILP fails to improve or produce valid.
        # But we want to use the ILP result.
        
        # Let's do a final check.
        # If we have time, run one final ILP on the accumulated active_indices.
        
        if time.time() < deadline - 5.0:
             prob = pulp.LpProblem("FinalCovering", pulp.LpMinimize)
             x = {}
             for idx in active_indices:
                 x[idx] = pulp.LpVariable(f"x_{idx}", cat=pulp.LpBinary)
             prob += pulp.lpSum(x[idx] for idx in active_indices)
             
             for i in range(n_t):
                 covering_blocks = [idx for idx in active_indices if i in block_to_covered[idx]]
                 if covering_blocks:
                     prob += pulp.lpSum(x[idx] for idx in covering_blocks) >= 1
             
             solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=5.0)
             prob.solve(solver)
             
             final_indices = []
             for idx in active_indices:
                 if x[idx].value() is not None and x[idx].value() > 0.5:
                     final_indices.append(idx)
             
             if final_indices:
                 return [all_blocks[i] for i in final_indices]
        
        # Fallback to greedy
        return greedy_cover(0)
        
    # 1. Get initial greedy solution
    best_blocks = greedy_cover(0)
    best_len = len(best_blocks)
    
    # 2. Try Column Generation
    cg_result = solve_column_generation(best_blocks)
    
    # Validate CG result
    if cg_result:
        covered = set()
        for block in cg_result:
            for ts in itertools.combinations(block, t):
                covered.add(t_subset_to_idx[ts])
        if len(covered) == n_t and len(cg_result) < best_len:
            best_len = len(cg_result)
            best_blocks = cg_result
    
    # 3. If time permits, try a few more greedy starts and maybe CG on the best
    for seed in range(1, 5):
        if time.time() > deadline - 15.0:
            break
        
        init = greedy_cover(seed)
        if len(init) < best_len:
            best_len = len(init)
            best_blocks = init
    
    # Final validation
    if best_blocks is None:
        return []
        
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
            
    if len(covered) != n_t:
        # Fallback to a valid greedy solution
        return greedy_cover(0)
        
    return best_blocks
