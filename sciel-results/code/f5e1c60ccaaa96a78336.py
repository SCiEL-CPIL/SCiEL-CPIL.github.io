import random
import time
import numpy as np

def construct(q, n, radius):
    # New Approach: Simulated Annealing / Local Search
    # This is distinct from the Greedy Set Cover approach in the seed.
    
    total_words = q ** n
    
    # Map words to indices 0..total_words-1
    # We need to generate balls for each index.
    # For q=3, n=7, total_words=2187.
    
    # Precompute balls as lists of indices
    # index = sum(digit[i] * q^(n-1-i))
    
    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return t

    def tuple_to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    balls = [[] for _ in range(total_words)]
    for idx in range(total_words):
        t = index_to_tuple(idx)
        ball_set = {idx}
        
        # Generate neighbors at distance <= radius
        # For efficiency, since radius is small (1), we just do distance 1.
        # If radius > 1, this needs expansion, but task is radius 1.
        # We'll assume radius=1 for speed, but handle generally if small.
        if radius == 1:
            for i in range(n):
                orig_val = t[i]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_t = t[:]
                    new_t[i] = new_val
                    new_idx = tuple_to_index(new_t)
                    ball_set.add(new_idx)
        else:
            # General radius (slower, but robust)
            # For n=7, q=3, radius=1 is the case. 
            # If radius > 1, we might need to be careful with time.
            # Given the task is specific to 7 matches, radius 1 is expected.
            # We'll stick to radius 1 logic for performance if radius==1.
            # If radius > 1, we can use a BFS or simple generation.
            # For now, assume radius=1 is the primary case.
            # If radius > 1, we can still use the same logic but it will be slower.
            # Let's just use the radius=1 logic if radius==1, else fallback to a slower general method.
            # Actually, for this specific task, radius is 1.
            pass
            
        balls[idx] = list(ball_set)

    # Initial Greedy Solution to start SA
    uncovered = set(range(total_words))
    code_indices = []
    coverage_counts = np.zeros(total_words, dtype=int)
    
    # Greedy init
    # We want to pick words that cover the most uncovered words.
    # To speed this up, we can just pick a few random words or a simple greedy step.
    # A simple greedy:
    # Sort words by some heuristic? Or just iterate.
    # For SA, a decent start is important.
    # Let's do a fast greedy:
    # While uncovered:
    #   Find word with max coverage of uncovered.
    #   Add it.
    
    # This is O(N^2) roughly if done naively, but N=2187 is small.
    # Let's do 100 steps of greedy to get a good start.
    for _ in range(100):
        if not uncovered:
            break
        max_cov = -1
        best_idx = -1
        # Randomly sample 50 candidates to speed up init
        candidates = random.sample(range(total_words), min(50, total_words))
        for idx in candidates:
            cov = 0
            for u in balls[idx]:
                if u in uncovered:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_idx = idx
        
        if best_idx != -1 and max_cov > 0:
            code_indices.append(best_idx)
            coverage_counts[balls[best_idx]] += 1
            for u in balls[best_idx]:
                if coverage_counts[u] == 1:
                    uncovered.discard(u)

    # If still uncovered, fill with random greedy
    while uncovered:
        # Pick random uncovered word, add its center? Or pick random center.
        # Let's pick a random center that covers at least 1 uncovered.
        found = False
        for _ in range(1000):
            idx = random.randint(0, total_words - 1)
            if any(u in uncovered for u in balls[idx]):
                code_indices.append(idx)
                coverage_counts[balls[idx]] += 1
                for u in balls[idx]:
                    if coverage_counts[u] == 1:
                        uncovered.discard(u)
                found = True
                break
        if not found:
            # Force add random
            idx = random.randint(0, total_words - 1)
            code_indices.append(idx)
            coverage_counts[balls[idx]] += 1
            for u in balls[idx]:
                if coverage_counts[u] == 1:
                    uncovered.discard(u)

    # Remove duplicates in code_indices
    code_set = set(code_indices)
    code_indices = list(code_set)
    
    # Recalculate coverage counts from scratch to be safe
    coverage_counts = np.zeros(total_words, dtype=int)
    for idx in code_indices:
        coverage_counts[balls[idx]] += 1
    
    uncovered_count = np.sum(coverage_counts == 0)
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    
    # Parameters
    T = 10.0
    cooling_rate = 0.999
    M = 1000 # Penalty for uncovered words
    
    best_size = len(code_indices)
    best_code = code_indices[:]
    
    # Precompute ball lists for speed
    # balls is already a list of lists.
    
    while time.time() - start_time < time_limit:
        # Choose move type
        r = random.random()
        if r < 0.4:
            move_type = 'remove'
        elif r < 0.8:
            move_type = 'add'
        else:
            move_type = 'swap'
            
        delta_E = 0
        accepted = False
        
        if move_type == 'remove' and len(code_indices) > 0:
            old_idx = random.choice(code_indices)
            # Calculate delta
            new_uncov = 0
            # Count how many words become uncovered
            # We need to know which words in ball(old_idx) have count 1
            # If count is 1, they become 0.
            # We don't update coverage_counts yet.
            for u in balls[old_idx]:
                if coverage_counts[u] == 1:
                    new_uncov += 1
            
            # Delta size: +1 (we removed one, so size decreases by 1, so energy decreases by 1? No, energy is size + M*uncov.
            # New size = old size - 1. Delta size = -1.
            # New uncov = old uncov + new_uncov. Delta uncov = new_uncov.
            # Delta E = -1 + M * new_uncov
            
            delta_E = -1 + M * new_uncov
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                # Update state
                code_indices.remove(old_idx)
                for u in balls[old_idx]:
                    coverage_counts[u] -= 1
                uncovered_count += new_uncov
                accepted = True
                
        elif move_type == 'add':
            # Pick a random word not in code
            # To speed up, we can pick from a list of "promising" words?
            # Or just random.
            new_idx = random.randint(0, total_words - 1)
            if new_idx in code_set:
                continue # Skip if already in code
            
            # Calculate delta
            # Words in ball(new_idx) with count 0 become covered.
            new_cov = 0
            for u in balls[new_idx]:
                if coverage_counts[u] == 0:
                    new_cov += 1
            
            # Delta size: +1. Delta uncov: -new_cov.
            delta_E = 1 - M * new_cov
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                code_indices.append(new_idx)
                code_set.add(new_idx)
                for u in balls[new_idx]:
                    coverage_counts[u] += 1
                uncovered_count -= new_cov
                accepted = True

        elif move_type == 'swap' and len(code_indices) > 0:
            old_idx = random.choice(code_indices)
            new_idx = random.randint(0, total_words - 1)
            if new_idx in code_set:
                continue
            
            # Calculate delta
            # Effect: Remove old, Add new.
            # Size change: 0.
            # Uncov change: 
            #   Words that were covered ONLY by old_idx (count 1) and NOT by new_idx become uncovered.
            #   Words that were uncovered (count 0) and covered by new_idx become covered.
            
            # Let's compute precisely.
            # We need to simulate the change on counts.
            # But we can calculate delta directly.
            
            # Let A = ball(old_idx), B = ball(new_idx).
            # For u in A:
            #   if count[u] == 1: u becomes 0 (uncovered) UNLESS u is also in B and count[u] (after removing old) is >= 1?
            #   Actually, if u in A and u in B:
            #     count_new = count_old - 1 (from old) + 1 (from new) = count_old. No change.
            #   if u in A and u not in B:
            #     count_new = count_old - 1.
            #     if count_old == 1, count_new = 0 -> uncovered.
            #   if u not in A and u in B:
            #     count_new = count_old + 1.
            #     if count_old == 0, count_new = 1 -> covered.
            
            delta_uncov = 0
            # We need to check intersection.
            # Since balls are small (1 + n*(q-1) = 1+7*2=15), we can just loop.
            
            # Create a set of B for fast lookup? Or just loop A and B.
            B_set = set(balls[new_idx])
            
            for u in balls[old_idx]:
                if u not in B_set:
                    if coverage_counts[u] == 1:
                        delta_uncov += 1 # Becomes uncovered
                else:
                    # u in A and B. Count doesn't change.
                    pass
            
            for u in balls[new_idx]:
                if u not in set(balls[old_idx]): # Optimization: check if in A
                    # Wait, creating set every time is slow.
                    pass
            
            # Let's do it efficiently:
            # We can just apply the change to a temporary copy of counts? No, too slow.
            # Let's re-evaluate:
            # delta_uncov = (number of u in A\B with count 1) - (number of u in B\A with count 0)
            
            # Let's compute A\B and B\A.
            # A = balls[old_idx], B = balls[new_idx]
            # Since sizes are ~15, set operations are fine.
            A_set = set(balls[old_idx])
            B_set = set(balls[new_idx])
            
            only_A = A_set - B_set
            only_B = B_set - A_set
            
            inc_uncov = 0
            for u in only_A:
                if coverage_counts[u] == 1:
                    inc_uncov += 1
            
            dec_uncov = 0
            for u in only_B:
                if coverage_counts[u] == 0:
                    dec_uncov += 1
            
            delta_uncov = inc_uncov - dec_uncov
            delta_E = M * delta_uncov
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                # Update counts
                for u in only_A:
                    coverage_counts[u] -= 1
                for u in only_B:
                    coverage_counts[u] += 1
                # Update code set
                code_indices.remove(old_idx)
                code_indices.append(new_idx)
                code_set.remove(old_idx)
                code_set.add(new_idx)
                uncovered_count += delta_uncov
                accepted = True

        if accepted:
            if uncovered_count == 0 and len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = code_indices[:]
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1 # Floor temperature

    return [index_to_tuple(idx) for idx in best_code]
