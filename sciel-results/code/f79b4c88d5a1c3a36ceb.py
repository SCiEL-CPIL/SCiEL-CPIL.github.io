"""Improved covering code search for K_3(6,1)."""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(q), repeat=n - 4)]
    
    # Precompute all words and their neighborhoods
    # Total space: 3^6 = 729 words
    all_words = np.array([[a, b, c, d, e, f] for a in range(3) for b in range(3) 
                          for c in range(3) for d in range(3) for e in range(3) for f in range(3)])
    total_words = 729
    
    # Precompute neighbors for each word (word itself + all words at distance 1)
    # Each word has 1 + 6*2 = 13 neighbors (including itself)
    neighbors = []
    for w in all_words:
        neigh = [w]
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    neigh.append(new_w)
        neighbors.append(neigh)
    
    # Convert neighbors to indices for fast lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbor_indices = []
    for neigh in neighbors:
        neighbor_indices.append([word_to_idx[tuple(w)] for w in neigh])
    
    # Convert to numpy arrays for vectorized operations
    neighbor_indices = np.array(neighbor_indices, dtype=np.int32)
    
    def coverage_check(codewords):
        """Check if all words are covered. Returns True if valid."""
        if len(codewords) == 0:
            return False
        covered = np.zeros(total_words, dtype=bool)
        cw = np.array(codewords, dtype=np.int32)
        for idx in cw:
            covered[neighbor_indices[idx]] = True
        return np.all(covered)
    
    def uncovered_count(codewords):
        """Return number of uncovered words."""
        if len(codewords) == 0:
            return total_words
        covered = np.zeros(total_words, dtype=bool)
        cw = np.array(codewords, dtype=np.int32)
        for idx in cw:
            covered[neighbor_indices[idx]] = True
        return np.sum(~covered)
    
    # Start with a good initial solution using greedy algorithm
    def greedy_construct():
        remaining = set(range(total_words))
        codewords = []
        while remaining:
            # Find the word that covers the most remaining words
            best_word = None
            best_count = -1
            for w in list(remaining):
                count = sum(1 for nb in neighbor_indices[w] if nb in remaining)
                if count > best_count:
                    best_count = count
                    best_word = w
            codewords.append(best_word)
            # Remove all covered words
            for nb in neighbor_indices[best_word]:
                remaining.discard(nb)
        return codewords
    
    # Initial greedy solution
    codewords = greedy_construct()
    best_solution = list(codewords)
    best_size = len(codewords)
    
    print(f"Initial greedy solution: {best_size} codewords")
    
    # Local search: try to remove redundant codewords
    def try_remove(codewords):
        """Try to remove each codeword if possible."""
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                # Check if removing codeword[i] still covers everything
                remaining = codewords[:i] + codewords[i+1:]
                if coverage_check(remaining):
                    codewords = remaining
                    improved = True
                    break
        return codewords
    
    # Try to improve by swapping
    def local_search(codewords, max_iters=1000):
        """Local search with swaps and removals."""
        codewords = list(codewords)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try removal
            removed = False
            for i in range(len(codewords)):
                remaining = codewords[:i] + codewords[i+1:]
                if coverage_check(remaining):
                    codewords = remaining
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try swap: replace one codeword with another
            if len(codewords) > 0:
                for i in range(len(codewords)):
                    # Try replacing codewords[i] with a random word
                    for _ in range(5):  # Try a few random replacements
                        new_word = random.randint(0, total_words - 1)
                        if new_word in codewords and new_word != codewords[i]:
                            continue
                        new_codewords = codewords[:i] + [new_word] + codewords[i+1:]
                        # Check if this covers everything
                        if coverage_check(new_codewords):
                            # Check if it covers more or same
                            codewords = new_codewords
                            break
                    if time.time() > deadline:
                        break
        return codewords
    
    # Run local search
    codewords = local_search(codewords, max_iters=5000)
    if len(codewords) < best_size:
        best_solution = list(codewords)
        best_size = len(codewords)
        print(f"Improved to {best_size} codewords")
    
    # Try multiple random restarts with different initial solutions
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create a random initial solution and improve it
        # Start with a subset of the greedy solution plus some random additions
        if best_size > 73:
            # Try to build from scratch with randomization
            remaining = set(range(total_words))
            codewords = []
            random.shuffle(list(remaining))
            while remaining and time.time() < deadline:
                # Pick a random word from remaining
                candidates = [w for w in random.sample(list(remaining), min(20, len(remaining)))]
                best_word = None
                best_count = -1
                for w in candidates:
                    count = sum(1 for nb in neighbor_indices[w] if nb in remaining)
                    if count > best_count:
                        best_count = count
                        best_word = w
                if best_word is None:
                    break
                codewords.append(best_word)
                for nb in neighbor_indices[best_word]:
                    remaining.discard(nb)
            
            if coverage_check(codewords):
                codewords = local_search(codewords, max_iters=1000)
                if len(codewords) < best_size:
                    best_solution = list(codewords)
                    best_size = len(codewords)
                    print(f"Restart {restart}: improved to {best_size} codewords")
    
    # Final polish: aggressive local search on best solution
    if time.time() < deadline - 5:
        best_solution = local_search(best_solution, max_iters=10000)
        best_size = len(best_solution)
    
    # Convert indices back to codewords
    result = [all_words[i].tolist() for i in best_solution]
    
    # Verify
    if not coverage_check(best_solution):
        # Fallback to safe solution
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=2)]
    
    return result
