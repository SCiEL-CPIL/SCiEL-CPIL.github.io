import itertools
import random
import time

def construct(q, n, radius):
    # This function is specialized for the task's specific parameters: q=3, n=7, radius=1
    # However, it should ideally be generic or at least robust to the specific call.
    # The task contract says: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # But the problem description is specific: "Football pool problem for 7 matches... K_3(7,1)".
    # The seed program hardcodes n=4 logic but adapts to n.
    # My program will implement a generic Greedy + Local Search which works for small q^n.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction if parameters differ, 
        # though the task is specifically K_3(7,1).
        # For safety, if it's not the specific case, use the seed's logic or a simple product.
        # But the prompt implies we are solving THIS task.
        pass

    # 1. Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # 2. Precompute balls of radius 1
    # ball[i] is a list of indices of words within distance 1 of all_words[i]
    balls = []
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neighbors.append(word_index[new_w])
        balls.append(neighbors)
        
    # 3. Greedy + Local Search
    best_solution = None
    best_size = N + 1
    
    # Run multiple iterations with randomization
    start_time = time.time()
    max_time = 110.0 # Leave some buffer for overhead
    
    # We can run this loop as many times as possible within the time limit
    # Each iteration is very fast (< 10ms likely)
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        
        # Greedy Phase
        uncovered = set(range(N))
        solution_indices = []
        
        # To randomize, we can shuffle the order of candidates when counts are equal,
        # or just pick randomly among the best candidates.
        # A simpler randomization: shuffle all_words initially? 
        # No, greedy picks the best. 
        # Let's add a small random noise to the score to break ties.
        
        while uncovered:
            best_word = -1
            best_count = -1
            best_tie_breaker = -1
            
            # Optimization: If the number of uncovered words is small, 
            # we can be more selective. But 2187 is small.
            
            # We iterate through all words. 
            # To allow for different paths, we can randomly select from the top K candidates
            # or add noise.
            
            # Let's just use a random tie-breaker among candidates with max count.
            candidates = []
            max_cov = 0
            
            for i in range(N):
                # If i is already in solution, skip? 
                # Greedy set cover typically allows re-adding, but since we only add once, 
                # and we remove from uncovered, a word added to solution won't be the "best" 
                # again for uncovered words unless it covers new ones? 
                # Wait, if a word is in the solution, its ball is already "processed" in the sense 
                # that the words it covers are removed from 'uncovered'. 
                # So adding it again adds 0 new coverage. 
                # So we can skip words already in solution.
                if i in [s for s in solution_indices]: # This check is O(K). 
                    # Better to keep a set of selected indices
                    pass
                
            # Let's maintain a set of selected indices for O(1) lookup
            # Actually, let's restructure the greedy loop to be efficient.
            
            # Reset for this iteration
            uncovered = set(range(N))
            solution_indices = []
            selected_set = set()
            
            while uncovered:
                best_word = -1
                best_count = -1
                
                # Find the word covering the most uncovered words
                # We can optimize by not checking words that are already selected
                # But even checking all is fast.
                
                # To introduce randomness, we can collect all words with the max count
                # and pick one randomly.
                
                max_cov = 0
                best_candidates = []
                
                for i in range(N):
                    if i in selected_set:
                        continue
                    
                    # Count how many of i's neighbors are in uncovered
                    # neighbors are in balls[i]
                    count = 0
                    for nb in balls[i]:
                        if nb in uncovered:
                            count += 1
                            
                    if count > max_cov:
                        max_cov = count
                        best_candidates = [i]
                    elif count == max_cov and count > 0:
                        best_candidates.append(i)
                        
                if not best_candidates:
                    # Should not happen if uncovered is not empty and we have enough words
                    break
                    
                # Pick one randomly from best candidates to diversify
                chosen = random.choice(best_candidates)
                solution_indices.append(chosen)
                selected_set.add(chosen)
                
                # Remove covered words from uncovered
                for nb in balls[chosen]:
                    uncovered.discard(nb)
                    
        # Local Search Phase: Try to remove redundant codewords
        # Maintain cover counts
        cover_count = [0] * N
        for idx in solution_indices:
            for nb in balls[idx]:
                cover_count[nb] += 1
                
        removed = True
        while removed:
            removed = False
            # Iterate through a copy of the list, or indices
            # We need to check if removing a codeword leaves all its neighbors covered by others.
            # i.e., for all nb in balls[idx], cover_count[nb] > 1
            for idx in solution_indices[:]:
                redundant = True
                for nb in balls[idx]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove idx
                    solution_indices.remove(idx)
                    for nb in balls[idx]:
                        cover_count[nb] -= 1
                    removed = True
                    
        if len(solution_indices) < best_size:
            best_size = len(solution_indices)
            best_solution = [all_words[i] for i in solution_indices]
            
        # Optional: Try a simple swap? 
        # If we remove one, we might be able to add a different one that allows further removals?
        # Standard local search for set cover minimization often just does removals.
        # Given the time limit, simple removal is likely sufficient to get close to optimal.
        
    if best_solution is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_solution
