import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Parameters for this specific task
    # q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed for unexpected inputs
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = q ** n
    powers = [q**i for i in range(n)]
    
    # Precompute neighbors (ball of radius 1)
    neighbors_list = []
    for i in range(N):
        nbrs = [i]
        for k in range(n):
            current_digit = (i // powers[k]) % q
            for v in range(q):
                if v == current_digit: continue
                new_i = i - current_digit * powers[k] + v * powers[k]
                nbrs.append(new_i)
        neighbors_list.append(nbrs)
    
    # Convert to numpy arrays for fast operations
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    # Precompute inverse neighbors: for each word, which codewords cover it
    # Since the graph is symmetric (distance <= 1 is symmetric), inverse neighbors are the same as neighbors
    # But we need to know, for a set of chosen codewords, which words they cover.
    
    def index_to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        return digits

    def run_criticality_greedy(random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        covered = np.zeros(N, dtype=bool)
        code = []
        code_set = set()
        
        # Precompute the list of coders for each word (neighbors)
        # neighbors[i] is the list of words covered by codeword i
        # We need: for word w, list of codewords c such that w in neighbors[c]
        # Since distance is symmetric, coders[w] = neighbors[w]
        
        # To speed up "find min criticality", we can maintain a count of available coders for each uncovered word.
        # Initially, every word has N available coders (all words in the space).
        # When we pick a codeword c, we remove it from the available pool.
        # This decreases the criticality of all words w that c covers.
        
        # However, updating criticality for all neighbors of c is O(17) per step.
        # Finding the min criticality among uncovered words is O(N) if done naively.
        # Total O(N * 17 * |Code|). |Code| ~ 500. N=6561. 6561*17*500 ~ 50M ops. Feasible in Python? Maybe slow.
        # Let's optimize.
        
        # Alternative: Just iterate through uncovered words to find min criticality.
        # Criticality(w) = number of c in neighbors[w] such that c not in code_set.
        
        while True:
            if time.time() > start_time + 105: # Time limit guard inside loop
                break
                
            # Find uncovered word with minimum criticality
            min_crit = N + 1
            target_word = -1
            
            # Optimization: If we just want a good heuristic, we can sample or check all.
            # Checking all 6561 words is fast enough if done efficiently.
            # But calculating criticality for each word involves checking its 17 neighbors.
            # 6561 * 17 ~ 100k operations per iteration.
            # 500 iterations * 100k = 50M operations. This might be too slow in pure Python loops.
            
            # Let's use a slightly different approach:
            # 1. Pick a random uncovered word w.
            # 2. Check its criticality.
            # 3. If it's low, proceed.
            # 4. Otherwise, keep looking.
            
            # Actually, a simpler and effective heuristic often used:
            # Standard Greedy: Maximize new coverage.
            # Criticality Greedy: 
            #   Find w_uncovered with min (count of available coders for w).
            #   Pick c in coders(w) that maximizes new coverage.
            
            # Let's try to vectorize the criticality calculation.
            # available_coders_mask is not easy to maintain as a bitset for N=6561.
            
            # Let's stick to a modified greedy that is fast:
            # 1. Compute marginal gain for all codewords not in code_set.
            # 2. Pick the one with max gain.
            # This is what the current program does.
            
            # To incorporate criticality, we can penalize codewords that cover few "hard" words?
            # Or, we can prioritize covering words with few neighbors.
            
            # Let's try a different strategy: 
            # "Worst-First" Greedy.
            # 1. Identify the uncovered word w with the fewest neighbors in the *remaining* candidate set?
            #    No, the candidate set is all words. The "available" coders are those not yet picked.
            
            # Let's implement a fast approximation of criticality greedy:
            # 1. Find the uncovered word w with the minimum number of neighbors *in the total space*? 
            #    No, that's constant (17).
            # 2. Find the uncovered word w with the minimum number of neighbors *that are still "good" candidates*?
            
            # Let's use the standard greedy but with a tie-breaker or a slight modification.
            # Actually, the most effective simple improvement is often just running standard greedy multiple times with random noise, which the current program does.
            
            # Let's look at the "Inspiration" code more closely. It uses bitsets.
            # Python integers can act as arbitrary precision bitsets.
            # N=6561 bits.
            
            # Let's rewrite using Python integers as bitsets for speed.
            pass # We will implement the bitset version below as it is much faster.

    # Bitset approach for speed
    # 1. Precompute neighbor bitsets for all N words.
    # 2. Uncovered set is a big integer.
    # 3. Greedy step:
    #    - For each candidate c (not in code), count bits in (Uncovered & NeighborMask[c]).
    #    - This is O(N) per step if we check all N. 6561 * 500 = 3M big int ops.
    #    - Big int ops on 6561 bits are fast in C (Python's int is C-backed).
    
    def run_bitset_greedy(random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        # Precompute masks
        masks = [0] * N
        for i in range(N):
            m = 0
            for j in neighbors_list[i]:
                m |= (1 << j)
            masks[i] = m
        
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > start_time + 105:
                break
                
            # Find best codeword
            best_c = -1
            best_gain = -1
            
            # To speed up, we can sample candidates or iterate all.
            # Iterating all 6561 is fine.
            # We can skip codewords already in code_set.
            
            # Optimization: If we maintain a list of "active" candidates, we can remove chosen ones.
            # But checking `if c not in code_set` is O(1).
            
            for c in range(N):
                if c in code_set:
                    continue
                # Count new words covered
                # gain = bit_count(uncovered & masks[c])
                # Python 3.10+ has int.bit_count()
                gain = (uncovered & masks[c]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain == 0:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~masks[best_c]
            
        return code

    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        # Rebuild masks for current code? No, we have global masks.
        # We need to check if removing c leaves any word uncovered.
        # A word is covered if it is in the union of masks of other codewords.
        
        # Efficient check:
        # For each c in code, check if there exists w in masks[c] such that w is NOT in union(masks[o] for o in code, o!=c).
        # This is expensive.
        
        # Alternative:
        # Maintain a "coverage count" array?
        # For each word w, count how many codewords in `code` cover it.
        # If we remove c, any w with count 1 becomes uncovered.
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbors_list[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < start_time + 110:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Check if any word is uniquely covered by c
                has_unique = False
                for w in neighbors_list[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                        
                if not has_unique:
                    # Safe to remove
                    code.pop(i)
                    for w in neighbors_list[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break # Restart loop after modification
                    
        return code

    def local_search_swap(code):
        """Try to replace one codeword with another to reduce size or improve structure"""
        # This is complex. Let's stick to removal first.
        # A simple swap: 
        # 1. Pick a codeword c with low unique coverage.
        # 2. Try replacing it with a random c' that covers all words uniquely covered by c.
        
        # Given time constraints, removal is the most reliable improvement.
        # The greedy usually produces a near-minimal set, so removals are few.
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_len = float('inf')
    
    # Run 1: Deterministic
    code = run_bitset_greedy(random_seed=42)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code.copy()
        
    # Run multiple times with noise? 
    # The bitset greedy is deterministic if we iterate 0..N-1.
    # To add randomness, we can shuffle the order of checking candidates or add noise to gain.
    
    # Let's try a few random restarts with slight perturbations.
    for run_id in range(1, 20):
        if time.time() > deadline:
            break
            
        # Perturbation: Randomly skip some candidates? Or random tie-breaking?
        # Let's just run it again. It's fast.
        # Actually, standard greedy is very stable. 
        # To get diversity, we can randomize the *initial* order or add random noise to the gain.
        
        def run_noisy_greedy(seed):
            np.random.seed(seed)
            random.seed(seed)
            
            masks = [0] * N
            for i in range(N):
                m = 0
                for j in neighbors_list[i]:
                    m |= (1 << j)
                masks[i] = m
            
            uncovered = (1 << N) - 1
            code = []
            code_set = set()
            
            while uncovered:
                if time.time() > deadline:
                    break
                best_c = -1
                best_gain = -1
                
                # Add small noise to gain to break ties and explore
                # But we must ensure we don't pick a bad codeword.
                # We can sample a subset of top candidates?
                # For now, just standard greedy with random tie-breaking.
                
                # To make it faster, we can pre-filter candidates.
                # But 6561 is small.
                
                for c in range(N):
                    if c in code_set:
                        continue
                    gain = (uncovered & masks[c]).bit_count()
                    # Add tiny noise to break ties
                    noise = random.uniform(0, 0.1)
                    total = gain + noise
                    
                    if total > best_gain:
                        best_gain = total
                        best_c = c
                
                if best_c == -1 or best_gain < 1: # If noise made it 0? No, gain is int.
                    break
                if (uncovered & masks[best_c]).bit_count() == 0:
                    break
                    
                code.append(best_c)
                code_set.add(best_c)
                uncovered &= ~masks[best_c]
                
            return code

        code = run_noisy_greedy(run_id)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
    # Local search polish
    if best_code is not None and time.time() < deadline:
        polished = local_search_remove(best_code.copy())
        if len(polished) < best_len:
            best_len = len(polished)
            best_code = polished
            
        # Try a few more removal passes?
        # local_search_remove already loops until no improvement.
        
    # Convert best_code to required format
    result = [index_to_digits(idx) for idx in best_code]
    return result
