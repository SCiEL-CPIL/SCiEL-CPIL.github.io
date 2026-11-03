import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Helper functions
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    total_words = q ** n
    
    # Precompute balls for all q^n words using numpy for speed
    # Ball of radius 1: word itself + all words at Hamming distance 1
    # For each position i, change digit to each of q-1 other values
    all_words = np.arange(total_words, dtype=np.int64)
    
    # Precompute ball neighbors as a list of arrays
    balls = []
    for w in all_words:
        w_list = int_to_list(int(w), n)
        neighbors = [w]
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    neighbors.append(list_to_int(new_word))
        balls.append(neighbors)
    
    # Convert to numpy arrays for faster iteration
    balls_np = [np.array(b, dtype=np.int64) for b in balls]
    
    # Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4 over {0,1,2}
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    code_list = list(code)
    
    # Build coverage using numpy
    def compute_coverage(code_set):
        """Compute coverage counts for all words."""
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            coverage[balls_np[c]] += 1
        return coverage
    
    def get_removable(coverage):
        """Find codewords that can be removed without breaking coverage."""
        if time.time() > deadline - 5:
            return None
        
        # A word is uniquely covered if coverage count is 1
        # Find all words with coverage == 1
        unique_words = np.where(coverage == 1)[0]
        
        if len(unique_words) == 0:
            # All words have coverage >= 2, any codeword can be removed
            return code.copy()
        
        # Find which codewords are essential (uniquely cover some word)
        essential = set()
        for w in unique_words:
            # Find which codeword covers w
            for c in code:
                if w in balls_np[c]:
                    essential.add(c)
                    break
        
        removable = code - essential
        if not removable:
            return None
        return removable
    
    def try_swap(c_to_remove, coverage):
        """Try to find a replacement for c_to_remove."""
        if time.time() > deadline - 5:
            return None
        
        # Find words uniquely covered by c_to_remove
        unique_words = []
        for w in balls_np[c_to_remove]:
            if coverage[w] == 1:
                unique_words.append(w)
        
        if not unique_words:
            # Can just remove
            return None
        
        # Find candidate replacements that cover all unique_words
        # Candidates are in the intersection of balls of all unique_words
        candidate_set = set(balls_np[unique_words[0]])
        for w in unique_words[1:]:
            candidate_set &= set(balls_np[w])
        
        # Filter candidates: must not be in code
        valid_candidates = [w for w in candidate_set if w not in code]
        
        if not valid_candidates:
            return None
        
        return random.choice(valid_candidates)
    
    def greedy_remove():
        """Aggressively remove redundant codewords."""
        if time.time() > deadline - 10:
            return
        
        coverage = compute_coverage(code)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            removable = get_removable(coverage)
            if removable:
                c = random.choice(list(removable))
                code.remove(c)
                # Update coverage
                coverage[balls_np[c]] -= 1
                improved = True
    
    def greedy_swap():
        """Try to swap codewords to enable further removals."""
        if time.time() > deadline - 10:
            return False
        
        coverage = compute_coverage(code)
        for c in list(code):
            w_new = try_swap(c, coverage)
            if w_new is not None:
                code.remove(c)
                code.add(w_new)
                # Update coverage
                coverage[balls_np[c]] -= 1
                coverage[balls_np[w_new]] += 1
                return True
        return False
    
    def simulated_annealing():
        """Run simulated annealing to escape local minima."""
        if time.time() > deadline - 15:
            return
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        best_code = set(code)
        best_size = len(code)
        
        while time.time() < deadline - 15 and T > T_min:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try removal
                coverage = compute_coverage(code)
                removable = get_removable(coverage)
                if removable:
                    c = random.choice(list(removable))
                    code.remove(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # No removal possible, try swap
                    c = random.choice(list(code))
                    w_new = try_swap(c, coverage)
                    if w_new is not None:
                        code.remove(c)
                        code.add(w_new)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
            else:
                # Try swap
                coverage = compute_coverage(code)
                c = random.choice(list(code))
                w_new = try_swap(c, coverage)
                if w_new is not None:
                    code.remove(c)
                    code.add(w_new)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # Try adding a random word (expansion)
                    w_new = random.randint(0, total_words - 1)
                    if w_new not in code:
                        code.add(w_new)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Restore best code found
        code.clear()
        code.update(best_code)
    
    # Main Loop: First greedy removal, then simulated annealing
    while time.time() < deadline - 20:
        # Greedy removal phase
        greedy_remove()
        
        # Try swaps
        if greedy_swap():
            continue
        
        # If no more greedy improvement, run simulated annealing
        if time.time() > deadline - 15:
            break
        
        simulated_annealing()
        
        # After SA, try greedy removal again
        greedy_remove()
        
        # Check if we can do any more swaps
        if not greedy_swap():
            # Try a few more SA iterations
            if time.time() < deadline - 15:
                simulated_annealing()
                greedy_remove()
                if not greedy_swap():
                    break
    
    # Final greedy removal pass
    while time.time() < deadline - 2:
        coverage = compute_coverage(code)
        removable = get_removable(coverage)
        if removable:
            c = random.choice(list(removable))
            code.remove(c)
        else:
            break
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result
