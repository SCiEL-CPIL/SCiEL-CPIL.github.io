import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave 5s margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triple indices for each block
    # Also precompute overlap matrix? 
    # Overlap[i, j] = number of triples shared by block i and block j.
    # This is C(7,3) = 35 triples per block.
    # We can compute this on the fly or precompute.
    # Precomputing overlap matrix: 17296 x 17296 matrix is ~1.2GB if float64. Too big.
    # We will compute gains and overlaps incrementally or use sparse representations.
    
    block_triple_indices = [None] * num_blocks
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
        block_triple_indices[i] = np.array(tris, dtype=np.int32)

    # Precompute which blocks cover which triples for fast lookup?
    # triple_to_blocks: list of lists of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        for tri_idx in block_triple_indices[i]:
            triple_to_blocks[tri_idx].append(i)
    
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small t
    # Actually, for repair, we need to find a block covering a specific triple.
    # triple_to_blocks[tri_idx] is a list of block indices.

    def is_valid(sol):
        """Check if solution covers all triples."""
        if not sol:
            return num_triples == 0
        # Use a set of covered triples
        covered = set()
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                covered.add(tri_idx)
        return len(covered) == num_triples

    def get_uncovered_triples(sol):
        """Return list of uncovered triple indices."""
        if not sol:
            return list(range(num_triples))
        covered = set()
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                covered.add(tri_idx)
        return [i for i in range(num_triples) if i not in covered]

    def greedy_initialization(seed):
        """Standard greedy set cover to get a valid starting point."""
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=np.int8)
        selected = []
        selected_set = set()
        
        # Precompute gains? No, update incrementally.
        # For speed, we iterate through all blocks each time. 17k blocks * ~80 steps = 1.4M ops. Fast enough.
        
        while any(uncovered) and time.time() < deadline - 10:
            best_gain = -1
            candidates = []
            
            # Optimization: Only consider blocks that cover at least 1 uncovered triple
            # We can't easily skip blocks without checking.
            # Let's just iterate.
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                # Calculate gain
                # Sum of uncovered[triple_indices]
                gain = np.sum(uncovered[block_triple_indices[i]])
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain <= 0:
                break
                
            chosen = rng.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)
            
            # Update uncovered
            uncovered[block_triple_indices[chosen]] = 0
            
        return selected

    def repair(sol, seed):
        """Ensure solution is valid by adding blocks for uncovered triples."""
        rng = random.Random(seed)
        sol = list(sol)
        sol_set = set(sol)
        
        # Check validity
        uncovered = get_uncovered_triples(sol)
        
        while uncovered:
            # Pick a random uncovered triple
            tri_idx = rng.choice(uncovered)
            # Find a block covering it
            candidates = triple_to_blocks[tri_idx]
            # Filter out already selected
            candidates = [b for b in candidates if b not in sol_set]
            if not candidates:
                # Should not happen if we have a valid covering design available
                break
            
            # Pick one that maximizes gain on remaining uncovered triples
            best_block = candidates[0]
            best_gain = -1
            for b in candidates:
                gain = np.sum(uncovered[block_triple_indices[b]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            sol.append(best_block)
            sol_set.add(best_block)
            
            # Update uncovered
            # We need to recompute uncovered or update incrementally.
            # For simplicity in repair (which is short), recompute is okay if few steps.
            # But let's update incrementally.
            covered_by_new = set(block_triple_indices[best_block])
            uncovered = [t for t in uncovered if t not in covered_by_new]
            
        return sol

    def calculate_redundancy(sol):
        """Calculate sum of (count^2) for all triples. Lower is better (less overlap)."""
        if not sol:
            return 0
        counts = np.zeros(num_triples, dtype=np.int32)
        for idx in sol:
            counts[block_triple_indices[idx]] += 1
        return np.sum(counts * counts)

    def genetic_algorithm(seed, time_limit):
        rng = random.Random(seed)
        
        # Initialize population
        pop_size = 50
        population = []
        for _ in range(pop_size):
            # Start with greedy solution, then mutate
            base = greedy_initialization(seed + rng.randint(0, 1000))
            # Mutate: remove random redundant blocks, then repair
            mutated = list(base)
            # Try to remove some blocks
            for _ in range(rng.randint(0, 5)):
                if mutated:
                    idx_to_remove = rng.randint(0, len(mutated) - 1)
                    b_idx = mutated[idx_to_remove]
                    # Check if redundant
                    # Simple check: is this block the only one covering any triple?
                    # This is expensive. Let's just remove and repair.
                    mutated.pop(idx_to_remove)
            mutated = repair(mutated, seed + rng.randint(0, 1000))
            population.append(mutated)
        
        # Evaluate initial population
        fitnesses = [len(ind) for ind in population]
        
        best_sol = min(population, key=len)
        best_len = len(best_sol)
        best_redundancy = calculate_redundancy(best_sol)
        
        gen = 0
        while time.time() < time_limit:
            gen += 1
            if time.time() > time_limit - 5:
                break
            
            # Selection: Tournament
            new_population = []
            for _ in range(pop_size):
                # Pick 3 random parents
                parents = rng.sample(population, 3)
                parent_fitnesses = [len(p) for p in parents]
                best_parent = parents[parent_fitnesses.index(min(parent_fitnesses))]
                new_population.append(list(best_parent))
            
            # Crossover and Mutation
            for i in range(len(new_population)):
                if i % 2 == 0 and i + 1 < len(new_population):
                    p1 = new_population[i]
                    p2 = new_population[i + 1]
                    # Uniform crossover
                    child1 = []
                    child2 = []
                    set1 = set(p1)
                    set2 = set(p2)
                    for b in set1.union(set2):
                        if b in set1 and b in set2:
                            child1.append(b)
                            child2.append(b)
                        elif b in set1:
                            if rng.random() < 0.5:
                                child1.append(b)
                            else:
                                child2.append(b)
                        else:
                            if rng.random() < 0.5:
                                child2.append(b)
                            else:
                                child1.append(b)
                    
                    # Repair children
                    child1 = repair(child1, seed + i)
                    child2 = repair(child2, seed + i + 1)
                    
                    new_population[i] = child1
                    new_population[i + 1] = child2
                else:
                    # Just repair (clone with repair)
                    new_population[i] = repair(new_population[i], seed + i)
            
            # Mutation: Swap-based mutation to reduce redundancy
            # Pick a random individual
            idx = rng.randint(0, len(new_population) - 1)
            ind = new_population[idx]
            
            # Calculate redundancy
            redundancy = calculate_redundancy(ind)
            
            # Try to swap a block with another to reduce redundancy
            if ind:
                b_idx = rng.randint(0, len(ind) - 1)
                b1 = ind[b_idx]
                
                # Find blocks that overlap significantly with b1
                # We want to replace b1 with a block that covers the "unique" triples of b1
                # but has less overlap with the rest of the solution.
                
                # Identify triples uniquely covered by b1 in this solution
                counts = np.zeros(num_triples, dtype=np.int32)
                for b in ind:
                    counts[block_triple_indices[b]] += 1
                
                unique_triples = []
                for tri_idx in block_triple_indices[b1]:
                    if counts[tri_idx] == 1:
                        unique_triples.append(tri_idx)
                
                if unique_triples:
                    # Find candidates that cover all unique_triples
                    candidates = set()
                    for ut in unique_triples:
                        candidates.intersection_update(triple_to_blocks[ut])
                    candidates.discard(b1)
                    
                    if candidates:
                        # Pick candidate with lowest overlap with rest of solution
                        rest = [b for j, b in enumerate(ind) if j != b_idx]
                        rest_set = set(rest)
                        
                        best_cand = None
                        best_overlap = float('inf')
                        
                        # Sample candidates if too many
                        cand_list = list(candidates)
                        if len(cand_list) > 20:
                            cand_list = rng.sample(cand_list, 20)
                        
                        for c in cand_list:
                            # Calculate overlap with rest
                            overlap = 0
                            c_tris = block_triple_indices[c]
                            for tri_idx in c_tris:
                                if counts[tri_idx] > 0: # Covered by someone (including b1, but b1 is removed)
                                    # We need overlap with REST.
                                    # counts includes b1.
                                    # So overlap_with_rest = counts[tri_idx] - (1 if tri_idx in b1_tris else 0)
                                    if tri_idx in block_triple_indices[b1]:
                                        if counts[tri_idx] > 1:
                                            overlap += counts[tri_idx] - 1
                                    else:
                                        overlap += counts[tri_idx]
                            
                            if overlap < best_overlap:
                                best_overlap = overlap
                                best_cand = c
                        
                        if best_cand is not None:
                            # Perform swap
                            ind[b_idx] = best_cand
                            # No need to repair as we ensured coverage of unique triples
                            # But just in case, we can check.
                            # Actually, the swap preserves coverage of unique triples.
                            # Other triples are covered by rest.
                            # So it remains valid.
                            new_population[idx] = ind
            
            # Update population
            population = new_population
            
            # Track best
            for ind in population:
                if len(ind) < best_len:
                    best_len = len(ind)
                    best_sol = list(ind)
                    best_redundancy = calculate_redundancy(best_sol)
                elif len(ind) == best_len:
                    # Prefer lower redundancy for future mutations?
                    # For now, just keep the first one found or random.
                    # Let's keep the one with lower redundancy if we want to refine.
                    # But the primary goal is size.
                    pass

        return best_sol

    # Main Loop
    best_sol = None
    best_size = float('inf')
    
    # Run GA with different seeds
    num_restarts = 3
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        print(f"GA Restart {i}", file=sys.stderr)
        
        # Allocate time for each restart
        # Total time ~145s. Preprocessing takes some.
        # Let's give each restart ~40s.
        time_limit_for_restart = start_time + 45 + i * 45
        
        if i == 0:
             time_limit_for_restart = start_time + 50
        elif i == 1:
             time_limit_for_restart = start_time + 95
        else:
             time_limit_for_restart = deadline - 5

        ga_sol = genetic_algorithm(100 + i, time_limit_for_restart)
        
        if ga_sol and len(ga_sol) < best_size:
            # Verify
            if is_valid(ga_sol):
                best_size = len(ga_sol)
                best_sol = ga_sol
                print(f"New best: {best_size}", file=sys.stderr)

    if best_sol is None:
        # Fallback to simple greedy
        best_sol = greedy_initialization(1)
        best_sol = repair(best_sol, 1)
        best_size = len(best_sol)

    return [all_blocks[i] for i in best_sol]
