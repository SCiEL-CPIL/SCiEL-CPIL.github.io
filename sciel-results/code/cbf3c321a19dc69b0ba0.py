"""Combined strategy: Greedy Construction + Redundancy Pruning for Ternary Covering Code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: finish by 110s, limit is 120s
    
    # Handle trivial or very small cases immediately if possible, 
    # but focus on the specific hard case q=3, n=6, radius=1.
    
    # If the problem is too large for greedy, fall back to direct sum
    # But for q=3, n=6, total_words=729, which is small enough for greedy.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Direct sum of tetracode if applicable, else all words (valid but huge)
        # For the specific task, this path is unlikely.
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            tails = list(itertools.product(range(3), repeat=n - 4))
            return [h + list(tail) for h in tetracode for tail in tails]
        else:
            # Generic fallback: return all words (guaranteed valid, size q^n)
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    total_words = q ** n
    sphere_size = 1 + n * (q - 1) # 13 for n=6, q=3
    
    # Precompute neighbors for all words in the space
    # neighbors[i] contains indices of all words within Hamming distance 1 from i
    neighbors = np.zeros((total_words, sphere_size), dtype=np.int32)
    q_powers = [q ** (n - 1 - i) for i in range(n)]
    
    for i in range(total_words):
        nbrs = [i]
        for pos in range(n):
            pow_q = q_powers[pos]
            current_val = (i // pow_q) % q
            for new_val in range(q):
                if new_val != current_val:
                    delta = (new_val - current_val) * pow_q
                    nbrs.append(i + delta)
        neighbors[i] = nbrs
    
    best_code_indices = None
    best_size = float('inf')
    
    # Phase 1: Greedy Set Cover with Random Restarts
    # We run multiple instances of greedy to find a good starting code.
    # Greedy picks the word that covers the most currently uncovered words.
    
    num_greedy_restarts = 20
    for r in range(num_greedy_restarts):
        if time.time() > deadline - 30.0: # Leave 30s for pruning phase
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        # Greedy loop
        while True:
            # Calculate gain: number of uncovered neighbors for each candidate word
            # We only care about uncovered words, so we mask covered ones
            # gains[i] = sum(covered[neighbors[i]] == False)
            gains = (~covered)[neighbors].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking to explore different solutions
            if len(candidates) > 0:
                best_idx = candidates[np.random.randint(len(candidates))]
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True
            else:
                break # Should not happen if max_gain > 0
        
        if covered.all():
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code_indices = code_indices[:]
                
    if best_code_indices is None:
        # Fallback if greedy failed (should not happen)
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Phase 2: Pruning Redundant Codewords
    # The greedy code might be slightly larger than optimal due to local minima.
    # We try to remove codewords that are redundant (their coverage is fully provided by others).
    
    current_code = best_code_indices[:]
    current_size = len(current_code)
    
    # Precompute coverage sets for the current code
    # For each codeword index in current_code, store the set of word indices it covers
    # Since total_words is 729, we can use bitsets or simple sets.
    # Using sets of ints is fast enough for ~75 codewords * 13 neighbors.
    
    def get_coverage_set(word_idx):
        return set(neighbors[word_idx])
    
    # Map word_index -> list of codeword_indices (in current_code) that cover it
    # This allows fast redundancy checks.
    
    def try_prune(code_list):
        """
        Attempts to prune the code_list by removing redundant codewords.
        Returns the pruned list.
        """
        active = [True] * len(code_list)
        # Build coverage map: word_idx -> set of active codeword indices covering it
        # To make removal efficient, we maintain counts or sets.
        # Given small size, we can just recompute or update incrementally.
        
        # For simplicity and robustness within time limits, we use a "check on demand" approach
        # with random ordering of removals.
        
        m = len(code_list)
        # coverage[word_idx] = set of indices in code_list that cover word_idx
        coverage = [set() for _ in range(total_words)]
        
        for i, w_idx in enumerate(code_list):
            for nbr in neighbors[w_idx]:
                coverage[nbr].add(i)
        
        removed_any = True
        # Multiple passes to remove cascading redundancies
        while removed_any and time.time() < deadline - 5.0:
            removed_any = False
            # Random order for this pass
            order = list(range(m))
            random.shuffle(order)
            
            for i in order:
                if not active[i]:
                    continue
                
                # Check if codeword i is redundant
                # It is redundant if for every word it covers, there is another active codeword covering it
                is_redundant = True
                for nbr in neighbors[code_list[i]]:
                    # Check if any other active codeword covers nbr
                    # coverage[nbr] contains all active codewords covering nbr
                    # We need to check if any j != i is in coverage[nbr] and active[j] is True
                    # Note: coverage[nbr] was built with all indices. We need to filter by active.
                    # To avoid O(m) check per neighbor, we can maintain counts.
                    # But m is small (~75). Let's just iterate.
                    found_other = False
                    for j in coverage[nbr]:
                        if j != i and active[j]:
                            found_other = True
                            break
                    if not found_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active[i] = False
                    removed_any = True
                    # Optional: Update coverage sets to remove i?
                    # Not strictly necessary for correctness of "found_other" check if we check active[j],
                    # but it might speed up subsequent checks if we prune the sets.
                    # For now, relying on active[j] check is O(m) per neighbor.
                    # 75 codewords * 13 neighbors * 75 checks = ~73k ops per removal. Very fast.
                    
        return [code_list[i] for i in range(m) if active[i]]
    
    # Run pruning with several random restarts on the best greedy code found
    # We don't shuffle the code list itself, but the removal order inside try_prune is random.
    # We might also try removing codewords in different initial orders.
    
    num_prune_restarts = 10
    for p in range(num_prune_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Create a copy to prune
        code_to_prune = current_code[:]
        pruned_code = try_prune(code_to_prune)
        
        if len(pruned_code) < best_size:
            best_size = len(pruned_code)
            best_code_indices = pruned_code[:]
            # If we found a smaller code, we can try to prune it again in subsequent restarts?
            # The current_code variable is updated, so next restart starts from the new best.
            # This allows iterative improvement.
            
    # Final sanity check: ensure best_code_indices is not None
    if best_code_indices is None:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Convert indices back to words
    result_words = []
    for idx in best_code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result_words.append(w)
        
    return result_words

# EVOLVE-BLOCK-END
