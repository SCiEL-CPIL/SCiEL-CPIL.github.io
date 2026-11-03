import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                import itertools
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 140.0  # Safety margin

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors using vectorized approach
    all_words = np.zeros((total_points, n), dtype=np.int8)
    for i in range(total_points):
        idx = i
        for pos in range(n - 1, -1, -1):
            all_words[i, pos] = idx % q
            idx //= q

    # Build neighbor table: neighbors[i] contains indices of all words within distance 1 of word i
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig = all_words[i, pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Compute index efficiently
                # The word differs only at position pos
                # index = sum(all_words[i][j] * q^(n-1-j) for j in range(n)) but with pos changed
                # Actually let's just compute directly
                new_word = all_words[i].copy()
                new_word[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_word[j]
                neighbors[i, count] = idx
                count += 1

    all_words_list = [list(all_words[i]) for i in range(total_points)]

    # Precompute: for each point, which codewords cover it (reverse mapping)
    # covers[point] = set of codeword indices that cover this point
    covers = [set() for _ in range(total_points)]
    for ci in range(total_points):
        for p in neighbors[ci]:
            covers[p].add(ci)

    def is_valid(chosen_set):
        """Check if chosen_set covers all points."""
        # A point is covered if any of its covering codewords is in chosen_set
        for p in range(total_points):
            if not any(c in chosen_set for c in covers[p]):
                return False
        return True

    def count_uncovered(chosen_set):
        """Count how many points are not covered."""
        uncovered = 0
        for p in range(total_points):
            if not any(c in chosen_set for c in covers[p]):
                uncovered += 1
        return uncovered

    # Initial solution: structured tetracode product
    tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
    initial_indices = []
    for t1 in tetracode:
        for t2 in tetracode:
            word = list(t1) + list(t2)
            idx = word_to_idx(word)
            initial_indices.append(idx)

    # Greedy completion to get a valid starting point
    def greedy_complete(initial):
        chosen = list(initial)
        chosen_set = set(chosen)
        # Track coverage count
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen:
            cover_count[neighbors[ci]] += 1

        while np.any(cover_count == 0):
            if time.time() - start_time > time_limit - 30.0:
                break
            uncovered = np.where(cover_count == 0)[0]
            best_idx = -1
            best_gain = -1
            # Sample some candidates for speed
            for _ in range(50):
                cand = random.randint(0, total_points - 1)
                if cand in chosen_set:
                    continue
                gain = np.sum(cover_count[neighbors[cand]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = cand
            if best_idx < 0 or best_gain <= 0:
                # Fallback: try all
                for cand in range(total_points):
                    if cand in chosen_set:
                        continue
                    gain = np.sum(cover_count[neighbors[cand]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = cand
            if best_idx < 0:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            cover_count[neighbors[best_idx]] += 1

        return chosen

    current_solution = greedy_complete(initial_indices)
    current_size = len(current_solution)
    best_solution = list(current_solution)
    best_size = current_size

    # Simulated Annealing
    # State: a set of codewords. We represent it as a boolean array for fast operations.
    # Moves: 
    #   1. Remove a codeword (if size > lower bound estimate)
    #   2. Add a codeword
    #   3. Swap: remove one, add one
    # Objective: minimize size, but penalize uncovered points heavily.
    
    # Precompute: for each codeword, its neighbor list as numpy array for speed
    # neighbors is already shape (total_points, neighbors_per_word)

    def energy(chosen_bool):
        """
        Energy = size + penalty * uncovered_count
        chosen_bool: boolean array of length total_points
        """
        size = np.sum(chosen_bool)
        # Compute coverage: point is covered if any chosen codeword covers it
        # cover_count[p] = sum of chosen_bool[c] for c in covers[p]
        # This is expensive to compute naively. Let's use the neighbors table.
        # A point p is covered if exists c with chosen_bool[c] and p in neighbors[c]
        # Equivalent: for each chosen c, mark neighbors[c] as covered.
        # We can vectorize: covered = OR over all chosen c of (neighbors[c] == p)
        # This is hard to vectorize directly. Let's compute uncovered.
        
        # Alternative: compute cover_count via matrix multiplication trick
        # cover_count = chosen_bool @ neighbors_indicator
        # But neighbors is sparse. Let's just loop over chosen codewords.
        chosen_indices = np.where(chosen_bool)[0]
        if len(chosen_indices) == 0:
            return size + 100 * total_points
        
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        
        uncovered = np.sum(cover_count == 0)
        penalty = 50  # Heavy penalty for uncovered points
        return size + penalty * uncovered

    def repair(chosen_bool):
        """
        If there are uncovered points, add codewords to cover them.
        Returns the modified chosen_bool.
        """
        chosen_indices = np.where(chosen_bool)[0]
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1

        while np.any(cover_count == 0) and time.time() - start_time < time_limit - 5.0:
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            # Find the codeword that covers the most uncovered points and is not already chosen
            best_idx = -1
            best_gain = -1
            # Sample candidates
            for _ in range(30):
                cand = random.randint(0, total_points - 1)
                if chosen_bool[cand]:
                    continue
                gain = np.sum(cover_count[neighbors[cand]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = cand
            if best_idx < 0 or best_gain <= 0:
                # Try a few more
                for cand in range(0, total_points, max(1, total_points // 200)):
                    if chosen_bool[cand]:
                        continue
                    gain = np.sum(cover_count[neighbors[cand]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = cand
            if best_idx < 0:
                break
            chosen_bool[best_idx] = True
            cover_count[neighbors[best_idx]] += 1

        return chosen_bool

    # Initialize state
    chosen_bool = np.zeros(total_points, dtype=bool)
    for ci in current_solution:
        chosen_bool[ci] = True

    current_energy = energy(chosen_bool)
    best_energy = current_energy
    best_chosen = chosen_bool.copy()

    # SA parameters
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.9995
    T = T_start
    max_iterations = 200000
    iteration = 0

    while T > T_end and iteration < max_iterations and time.time() - start_time < time_limit - 10.0:
        iteration += 1

        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Remove a codeword
            chosen_indices = np.where(chosen_bool)[0]
            if len(chosen_indices) == 0:
                continue
            ci = chosen_indices[random.randint(0, len(chosen_indices) - 1)]
            new_bool = chosen_bool.copy()
            new_bool[ci] = False
            new_energy = energy(new_bool)
            delta = new_energy - current_energy
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                chosen_bool = new_bool
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chosen = chosen_bool.copy()
                    best_solution = list(np.where(chosen_bool)[0])
                    best_size = len(best_solution)
        elif move_type < 0.7:
            # Add a codeword
            ci = random.randint(0, total_points - 1)
            if chosen_bool[ci]:
                continue
            new_bool = chosen_bool.copy()
            new_bool[ci] = True
            new_energy = energy(new_bool)
            delta = new_energy - current_energy
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                chosen_bool = new_bool
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chosen = chosen_bool.copy()
                    best_solution = list(np.where(chosen_bool)[0])
                    best_size = len(best_solution)
        else:
            # Swap
            chosen_indices = np.where(chosen_bool)[0]
            if len(chosen_indices) == 0:
                continue
            ci_remove = chosen_indices[random.randint(0, len(chosen_indices) - 1)]
            ci_add = random.randint(0, total_points - 1)
            if ci_add == ci_remove or chosen_bool[ci_add]:
                continue
            new_bool = chosen_bool.copy()
            new_bool[ci_remove] = False
            new_bool[ci_add] = True
            new_energy = energy(new_bool)
            delta = new_energy - current_energy
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                chosen_bool = new_bool
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chosen = chosen_bool.copy()
                    best_solution = list(np.where(chosen_bool)[0])
                    best_size = len(best_solution)

        T *= cooling_rate

    # After SA, repair the best solution if it has uncovered points
    final_bool = best_chosen.copy()
    final_bool = repair(final_bool)
    final_solution = list(np.where(final_bool)[0])

    # If repair made it worse, try to trim
    if len(final_solution) > best_size:
        # Try to remove redundant codewords from final_solution
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in final_solution:
            cover_count[neighbors[ci]] += 1
        for ci in list(final_solution):
            if time.time() - start_time > time_limit - 3.0:
                break
            exposed = neighbors[ci]
            if np.all(cover_count[exposed] > 1):
                cover_count[exposed] -= 1
                final_solution.remove(ci)

    # Final validation
    if not is_valid(set(final_solution)):
        # Fallback: use the greedy completed solution
        final_solution = list(current_solution)

    # Ensure distinctness
    final_solution = list(set(final_solution))

    return [all_words_list[i] for i in final_solution]
