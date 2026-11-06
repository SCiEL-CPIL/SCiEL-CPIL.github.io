import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute "partner" indices for each position in each AP
    # For each AP (i, j, k, l), if we flip i, we need to know j, k, l.
    # We can represent the coloring as a bitset or an array of 0/1.
    # A 4-AP is monochromatic if all 4 bits are equal.
    # Let's precompute the indices of the other 3 elements for each of the 4 positions in each AP.
    # pos0_partners[ap_idx] = (j, k, l)
    # pos1_partners[ap_idx] = (i, k, l)
    # pos2_partners[ap_idx] = (i, j, l)
    # pos3_partners[ap_idx] = (i, j, k)
    
    # Actually, a more efficient way for local search:
    # For each index i in 0..N-1, store the list of (other1, other2, other3) for each AP containing i.
    # This allows computing the delta of flipping i in O(degree(i)).
    
    index_to_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        a, b, c, d = aps[ap_idx]
        index_to_aps[a].append((b, c, d))
        index_to_aps[b].append((a, c, d))
        index_to_aps[c].append((a, b, d))
        index_to_aps[d].append((a, b, c))
        
    # Convert to numpy arrays for vectorized access
    # We will store for each index i, arrays of the other indices.
    # Since the number of APs per index varies, we'll just keep lists of tuples for simplicity 
    # but access them via numpy where possible in the inner loop? 
    # Actually, for N=256, the degree is small enough that pure Python loops with numpy scalar ops might be slow.
    # Let's try to vectorize the "flip all beneficial" step.
    
    # Alternative Strategy: 
    # Instead of greedy single flips, we can use a "wave" approach or simply a very fast simulated annealing
    # with a better initial state.
    
    # However, the prompt asks for a MATERIALLY DIFFERENT strategy.
    # Current: Multi-start Greedy/SA with random init.
    # New: Hierarchical Deterministic Construction + Fast Local Search.
    
    # Step A: Hierarchical Construction
    # 1. Start with a random bitstring.
    # 2. Optimize using a fast local search that evaluates ALL possible flips at once using vectorized operations?
    #    No, the dependencies are complex.
    # 3. Let's use a "Block" based approach. Divide [0, 256) into blocks of size B.
    #    Optimize one block at a time, holding others fixed. This reduces the problem size for the search.
    
    def compute_score(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)
        
    def local_search_block(c, block_start, block_end, max_iters=1000):
        """
        Perform local search on a specific block of indices [block_start, block_end).
        Indices outside the block are fixed.
        We iterate through indices in the block and flip if it improves the score.
        We repeat until no improvement or max_iters.
        """
        indices = np.arange(block_start, block_end)
        current_score = compute_score(c)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            improved = False
            # Shuffle order to avoid bias
            rng = np.random.default_rng()
            perm = rng.permutation(indices)
            
            for i in perm:
                # Compute delta for flipping i
                # Find all APs containing i
                # This is the bottleneck. We need fast access to partners.
                pass # Placeholder, see below for optimized version
        
        return current_score, c

    # The "local_search_block" above is pseudo-code. The bottleneck is finding partners.
    # Let's precompute the partner lists as numpy arrays of fixed size with padding, or just lists.
    # Given N=256, max APs per index is roughly 256/3 * 3 ~ 256? No, less.
    # Let's just use the list of tuples.
    
    def get_partners(i):
        return index_to_aps[i]
        
    def fast_local_search(c, max_sweeps=50):
        """
        Standard greedy local search, but optimized.
        """
        current_score = compute_score(c)
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 2:
                break
            any_flip = False
            # Fixed order is fine for greedy, but random order can help escape plateaus if we allow equal deltas?
            # Strict improvement only.
            for i in range(N):
                partners = index_to_aps[i]
                if not partners:
                    continue
                
                val_i = c[i]
                new_val_i = 1 - val_i
                delta = 0
                
                for p1, p2, p3 in partners:
                    v1, v2, v3 = c[p1], c[p2], c[p3]
                    # Mono before: v1==v2==v3==val_i
                    # Mono after: v1==v2==v3==new_val_i
                    if v1 == v2 and v2 == v3:
                        if v1 == val_i:
                            delta -= 1 # Was mono, won't be
                        else:
                            delta += 1 # Wasn't mono, will be
                
                if delta < 0:
                    c[i] = new_val_i
                    current_score += delta
                    any_flip = True
                    
            if not any_flip:
                break
                
        return current_score, c

    best_score = np.inf
    best_coloring = None
    
    # Strategy: Multi-scale Hierarchical Construction
    # 1. Initialize with a random coloring.
    # 2. Run global local search.
    # 3. "Freeze" large blocks and optimize inside? No, that's not better than global.
    # 4. Instead, let's use a "Coarse-to-Fine" approach on the structure of the problem?
    #    The problem is defined on [1, 256].
    #    Maybe we can construct a coloring that is periodic or has specific symmetry?
    #    Random is usually best for these.
    
    # Let's try a different local search: **Simulated Annealing with Fast Restart**
    # But the current program already does multi-start greedy.
    # What if we do **Parallel** local search? Python GIL prevents true parallelism for CPU-bound numpy unless using threads carefully, 
    # but numpy releases GIL. However, for N=256, the overhead of threading might be high.
    
    # Let's look at the **Structure** of 4-APs.
    # A 4-AP is determined by (a, d).
    # Total APs: sum_{d=1}^{85} (256 - 3d) = 85*256 - 3*(85*86/2) = 21760 - 8715 = 13045.
    # This is a small number.
    
    # What if we formulate this as a **Max-Cut** or **Ising Model** problem?
    # Minimize sum_{AP} [x_a x_b x_c x_d + (1-x_a)(1-x_b)(1-x_c)(1-x_d)]
    # This is a degree-4 polynomial.
    
    # Let's try a **Genetic Algorithm** approach?
    # Population of colorings, crossover, mutation, selection.
    # This is a different design family from local search.
    
    def genetic_algorithm(pop_size=50, generations=100):
        pop = [np.random.randint(0, 2, size=N, dtype=np.int8) for _ in range(pop_size)]
        scores = [compute_score(ind) for ind in pop]
        
        best_score_ga = min(scores)
        best_ind_ga = pop[scores.index(best_score_ga)].copy()
        
        for gen in range(generations):
            if time.time() > deadline - 5:
                break
            
            new_pop = []
            # Elitism: keep top 5
            sorted_indices = np.argsort(scores)
            for i in range(5):
                new_pop.append(pop[sorted_indices[i]].copy())
            
            # Crossover and Mutation
            while len(new_pop) < pop_size:
                # Tournament selection
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                if scores[i1] < scores[i2]:
                    parent1 = pop[i1]
                else:
                    parent1 = pop[i2]
                
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                if scores[i1] < scores[i2]:
                    parent2 = pop[i1]
                else:
                    parent2 = pop[i2]
                
                # Uniform crossover
                mask = np.random.randint(0, 2, size=N, dtype=bool)
                child = np.where(mask, parent1, parent2)
                
                # Mutation
                # Flip a few bits
                num_mutations = np.random.randint(1, 5)
                mut_indices = np.random.choice(N, num_mutations, replace=False)
                child[mut_indices] = 1 - child[mut_indices]
                
                new_pop.append(child)
            
            pop = new_pop
            scores = [compute_score(ind) for ind in pop]
            
            if min(scores) < best_score_ga:
                best_score_ga = min(scores)
                best_ind_ga = pop[scores.index(best_score_ga)].copy()
                
        return best_score_ga, best_ind_ga

    # Run GA
    ga_score, ga_coloring = genetic_algorithm()
    if ga_score < best_score:
        best_score = ga_score
        best_coloring = ga_coloring.copy()
        
    # Now, take the best GA result and polish it with Local Search
    if best_coloring is not None:
        final_score, final_coloring = fast_local_search(best_coloring.copy())
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_coloring.copy()
            
    # Also try multi-start local search from random starts, but fewer restarts since GA explored the space
    # We have time left?
    # Let's do a few more random restarts of local search to be safe.
    
    rng = np.random.default_rng()
    for r in range(50):
        if time.time() > deadline - 5:
            break
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        score, c_final = fast_local_search(c)
        if score < best_score:
            best_score = score
            best_coloring = c_final.copy()
            
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
