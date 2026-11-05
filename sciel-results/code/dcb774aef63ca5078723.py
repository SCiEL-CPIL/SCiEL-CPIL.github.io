import random
import time
import math

def construct(q, n, radius):
    # This implementation uses Simulated Annealing to find the minimum number of codewords.
    # It is designed for q=3, n=6, radius=1 but falls back to a greedy method for other parameters.
    
    if q != 3 or n != 6 or radius != 1:
        return _construct_fallback(q, n, radius)

    total_words = q ** n  # 729
    indices = list(range(total_words))
    
    def tuple_to_index(t):
        idx = 0
        for digit in t:
            idx = idx * q + digit
        return idx

    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute ball masks for all 729 words
    # ball_masks[i] is a bitmask of all words within Hamming distance 'radius' of word i
    ball_masks = [0] * total_words
    for i in indices:
        w = index_to_tuple(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = tuple_to_index(tuple(new_w))
                    mask |= (1 << j)
        ball_masks[i] = mask

    all_bits = (1 << total_words) - 1
    
    def calculate_energy(codewords):
        """Calculate the number of uncovered words."""
        covered = 0
        for c in codewords:
            covered |= ball_masks[c]
        return (all_bits & ~covered).bit_count()

    def simulated_annealing(K, time_limit):
        """
        Try to find a valid set of K codewords using Simulated Annealing.
        Returns a list of K indices if found, else None.
        """
        if K <= 0:
            return []
        
        # Initialize with a random set of K distinct codewords
        # Using a set for O(1) membership check and distinctness
        current_set = set(random.sample(indices, K))
        current_energy = calculate_energy(current_set)
        
        if current_energy == 0:
            return list(current_set)

        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        start_time = time.time()
        
        while T > T_min and (time.time() - start_time) < time_limit:
            # Pick a random element to remove and a random new element to add
            # We maintain a list for easy random access, but use set for membership
            # To make it efficient, we keep a list version in sync
            # Actually, for K~75, list operations are fine.
            # Let's use a list for the current state to allow O(1) swap by index
            pass 
            
        # The above loop structure is tricky with sets. Let's use a list.
        
        # Restart SA with list-based state
        current_list = random.sample(indices, K)
        current_set = set(current_list)
        current_energy = calculate_energy(current_list)
        
        if current_energy == 0:
            return current_list

        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        start_time = time.time()
        
        while T > T_min and (time.time() - start_time) < time_limit:
            # Generate a neighbor
            idx_remove = random.randint(0, K - 1)
            old_word = current_list[idx_remove]
            
            # Pick a random new word not in the current set
            # Simple rejection sampling is fine for K=75, N=729
            new_word = random.randint(0, total_words - 1)
            if new_word in current_set:
                # Retry a few times
                for _ in range(10):
                    new_word = random.randint(0, total_words - 1)
                    if new_word not in current_set:
                        break
                else:
                    # If we can't find a unique word (unlikely), skip
                    T *= cooling_rate
                    continue

            # Calculate new energy
            # Optimization: Instead of recalculating from scratch, we can update incrementally?
            # Energy = Uncovered.
            # Uncovered = Total - Covered.
            # Removing old_word: Covered might decrease.
            # Adding new_word: Covered might increase.
            # It's complex to update incrementally without bitset arithmetic on the union.
            # Given N=729, bit_count is very fast. Recalculating is O(K) bitwise ORs.
            # K=75. 75 ORs per step. This is very fast in Python if using big ints.
            
            # Let's try incremental calculation to be safe on time.
            # However, Python big int OR is implemented in C and is very fast.
            # Let's stick to full recalculation for simplicity and correctness, 
            # but optimize the inner loop.
            
            # Actually, let's just recalculate.
            # To speed up, we can pre-compute the union of all *except* the removed one?
            # That's expensive to maintain.
            
            # Let's just do the simple way first and see if it's fast enough.
            # 120 seconds limit.
            
            new_list = current_list[:]
            new_list[idx_remove] = new_word
            new_set = current_set.copy()
            new_set.remove(old_word)
            new_set.add(new_word)
            
            # Calculate energy
            covered = 0
            for c in new_list:
                covered |= ball_masks[c]
            new_energy = (all_bits & ~covered).bit_count()
            
            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_list = new_list
                current_set = new_set
                current_energy = new_energy
                if current_energy == 0:
                    return current_list
            
            T *= cooling_rate
        
        return None

    def greedy_initial():
        """Generate a valid set using greedy to get an upper bound."""
        uncovered = all_bits
        codewords = []
        while uncovered:
            max_cov = -1
            best_idx = -1
            for i in indices:
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return codewords

    # Get an initial upper bound
    init_codewords = greedy_initial()
    best_size = len(init_codewords)
    best_solution = init_codewords[:]
    
    start_time = time.time()
    time_limit = 115
    
    # Try to reduce the size
    # We know the lower bound is 71.
    # We start from best_size - 1 and go down.
    # If we find a valid set of size K, we keep it and try K-1.
    
    current_K = best_size
    
    while current_K > 70 and (time.time() - start_time) < time_limit:
        # Allocate a portion of the remaining time for this K
        # We don't know how many steps we have, so we just run SA for a fixed small duration or until time runs out?
        # Better: Run SA for a fixed number of iterations or a small time slice.
        # If we find a solution, great. If not, we might have wasted time.
        # Strategy: Try K. If success, best_size = K, current_K = K - 1.
        # If fail, break? Or try next lower K? 
        # Usually, if K is hard, K-1 is harder. So if we fail on K, we should probably stop.
        # But maybe SA just got stuck. 
        # Let's give each K a limited time budget, e.g., 5 seconds.
        
        time_budget = 5.0
        start_k_time = time.time()
        
        # Run SA multiple times if time permits?
        # For simplicity, one long SA run per K.
        solution = simulated_annealing(current_K, time_budget)
        
        if solution is not None:
            best_size = current_K
            best_solution = solution
            current_K -= 1
        else:
            # If we failed to find a valid set of size current_K, 
            # it's very likely current_K-1 is also not found quickly.
            # However, it's possible current_K is not minimal but SA failed.
            # Given the gap between 73 and 71 is small, and 73 is best known,
            # if we are at 73 and fail to go to 72, we stop.
            # If we are at 74 and fail to go to 73, we stop.
            break

    result = [list(index_to_tuple(i)) for i in best_solution]
    return result

def _construct_fallback(q, n, radius):
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1:
            break
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
    return [list(c) for c in codewords]
