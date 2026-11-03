import random
import time
import numpy as np

def construct(q, n, radius):
    # Problem is fixed to q=3, n=7, radius=1 based on task description, 
    # but the interface passes these values. 
    # The logic below is generic for q, n, radius=1 but optimized for the specific case.
    
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    total_words = q ** n
    # Map word tuple to index and vice versa
    # For q=3, n=7, total_words=2187. 
    # Using integer representation for speed.
    
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute cover matrix: cover_matrix[i, j] = True if word j is within radius of word i
    # This is a symmetric matrix for radius 1
    cover_matrix = np.zeros((total_words, total_words), dtype=bool)
    
    # Generate all words
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    for i in range(total_words):
        w = all_words[i]
        cover_matrix[i, i] = True  # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = tuple_to_idx(new_w)
                cover_matrix[i, j] = True
    
    # Balls as lists for local search
    balls = [np.where(cover_matrix[i])[0].tolist() for i in range(total_words)]
                
    # Greedy Set Cover
    best_solution = None
    
    def minimize_solution(solution):
        """Remove redundant codewords using cover counts."""
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in solution:
            cover_count[cover_matrix[c]] += 1
            
        changed = True
        current_sol = list(solution)
        
        while changed:
            changed = False
            for c in list(current_sol):
                # Check if c is redundant: all words it covers are covered by others
                if not np.any(cover_count[cover_matrix[c]] == 1):
                    # Remove c
                    current_sol.remove(c)
                    cover_count[cover_matrix[c]] -= 1
                    changed = True
        return current_sol

    def local_search_swap(solution, max_iterations=200):
        """Try to improve by swapping codewords to reduce size."""
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in solution:
            cover_count[cover_matrix[c]] += 1
        
        current_sol = list(solution)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            # Try removing each codeword and see if we can add a better one
            for i, c in enumerate(list(current_sol)):
                if time.time() > deadline:
                    break
                # Tentatively remove c
                cover_count[cover_matrix[c]] -= 1
                # Find uncovered words
                uncovered_mask = (cover_count == 0)
                if not np.any(uncovered_mask):
                    # Can remove c entirely
                    current_sol.pop(i)
                    improved = True
                    break
                # Find best replacement that covers all uncovered words
                uncovered_words = np.where(uncovered_mask)[0]
                # For each candidate, check if it covers all uncovered words
                # We need a word that covers all uncovered words
                # This is expensive, so we limit candidates
                best_candidate = -1
                best_gain = -1
                # Sample some candidates for speed
                n_candidates = min(50, total_words)
                candidates = np.random.choice(total_words, n_candidates, replace=False)
                for cand in candidates:
                    # Check if cand covers all uncovered words
                    if np.all(cover_matrix[cand, uncovered_words]):
                        # Count how many words it additionally covers (beyond uncovered)
                        gain = np.sum(cover_matrix[cand])
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = cand
                if best_candidate >= 0:
                    # Add the best candidate
                    current_sol[i] = best_candidate
                    cover_count[cover_matrix[best_candidate]] += 1
                    improved = True
                    break
                else:
                    # Restore c
                    cover_count[cover_matrix[c]] += 1
        
        return current_sol

    def greedy_run(random_seed):
        random.seed(random_seed)
        rng = np.random.RandomState(random_seed)
        uncovered = np.ones(total_words, dtype=bool)
        gains = np.array([len(balls[i]) for i in range(total_words)], dtype=np.int32)
        selected = []
        
        remaining = total_words
        
        while remaining > 0:
            # Find max gain
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one from candidates randomly
            idx = rng.choice(candidates)
            
            selected.append(idx)
            
            # Update gains and uncovered
            newly_covered = cover_matrix[idx] & uncovered
            uncovered[cover_matrix[idx]] = False
            remaining -= np.sum(newly_covered)
            
            # Update gains: for each newly covered word, decrement gain of all codewords that cover it
            if np.any(newly_covered):
                # For each newly covered word w, find all codewords j that cover w and decrement gains[j]
                for w in np.where(newly_covered)[0]:
                    gains[cover_matrix[:, w]] -= 1
        
        return selected

    # Run several greedy instances with local search
    run_count = 0
    max_runs = 100
    
    while run_count < max_runs:
        if time.time() > deadline - 30:  # Leave time for final polish
            break
        
        sol = greedy_run(run_count)
        sol = minimize_solution(sol)
        sol = local_search_swap(sol)
        
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = sol
            # Try more local search on the best solution
            best_solution = local_search_swap(best_solution, max_iterations=500)
        
        run_count += 1
    
    # Final polish: try more aggressive local search
    if best_solution is not None and time.time() < deadline - 5:
        best_solution = local_search_swap(best_solution, max_iterations=1000)
        best_solution = minimize_solution(best_solution)
    
    if best_solution is None:
        # Fallback to a trivial valid solution if something went wrong
        # e.g. all words
        return [idx_to_tuple(i) for i in range(total_words)]
        
    # Convert indices back to tuples
    return [idx_to_tuple(i) for i in best_solution]
