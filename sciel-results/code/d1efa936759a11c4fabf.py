import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Build cover matrix: cover_matrix[i, j] = 1 if word j is within distance <= radius of word i
    # For radius=1, this includes the word itself and all words at Hamming distance 1
    cover_matrix = np.zeros((total_space, total_space), dtype=np.uint8)
    for i in range(total_space):
        cover_matrix[i, i] = 1
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                cover_matrix[i, j] = 1

    # Precompute neighbor lists for each word (for faster operations)
    neighbor_lists = []
    for i in range(total_space):
        neighbors = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                neighbors.append(j)
        neighbor_lists.append(neighbors)

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords from a valid covering set."""
        selected = list(selected)
        if not selected:
            return selected

        # Build coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[cover_matrix[w]] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                # Check if all words covered by w have count > 1
                if np.all(counts[cover_matrix[w]] > 1):
                    counts[cover_matrix[w]] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    # Simulated Annealing
    def run_simulated_annealing(initial_selection, time_limit):
        """
        SA over binary selection vector.
        Energy = heavy_penalty * uncovered_count + light_penalty * selected_count
        Moves: add a random word, remove a random word, or swap
        """
        # Start from initial selection
        selection = np.zeros(total_space, dtype=bool)
        for w in initial_selection:
            selection[w] = True

        # Compute initial coverage
        covered = np.zeros(total_space, dtype=np.uint8)
        for w in range(total_space):
            if selection[w]:
                covered[cover_matrix[w]] = 1

        uncovered_count = total_space - np.sum(covered)
        selected_count = np.sum(selection)

        # Energy parameters
        heavy_penalty = 100.0
        light_penalty = 1.0

        def compute_energy(uncovered, selected):
            return heavy_penalty * uncovered + light_penalty * selected

        current_energy = compute_energy(uncovered_count, selected_count)
        best_energy = current_energy
        best_selection = selection.copy()

        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        cooling_rate = 0.995

        T = T_start
        step = 0

        while T > T_end and time.time() < time_limit:
            # Choose a random move
            move_type = random.random()
            if move_type < 0.3:
                # Add a random word
                w = random.randint(0, total_space - 1)
                if not selection[w]:
                    # Try adding
                    selection[w] = True
                    new_covered = covered.copy()
                    new_covered[cover_matrix[w]] = 1
                    new_uncovered = total_space - np.sum(new_covered)
                    new_selected = selected_count + 1
                    new_energy = compute_energy(new_uncovered, new_selected)
                    delta = new_energy - current_energy

                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        covered = new_covered
                        uncovered_count = new_uncovered
                        selected_count = new_selected
                        current_energy = new_energy

                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_selection = selection.copy()
                    else:
                        selection[w] = False

            elif move_type < 0.6:
                # Remove a random selected word
                selected_indices = np.where(selection)[0]
                if len(selected_indices) > 0:
                    w = selected_indices[random.randint(0, len(selected_indices) - 1)]
                    # Check if removal is safe
                    if np.all(covered[cover_matrix[w]] > 1) or random.random() < 0.1:
                        # Compute what would happen
                        new_covered = covered.copy()
                        # Subtract coverage of w, but keep words covered by others
                        for pos in cover_matrix[w]:
                            if covered[pos] == 1:
                                # Check if any other selected word covers pos
                                # This is expensive, so use a heuristic
                                pass

                        # Simpler: just try and compute
                        selection[w] = False
                        new_covered = np.zeros(total_space, dtype=np.uint8)
                        for idx in range(total_space):
                            if selection[idx]:
                                new_covered[cover_matrix[idx]] = 1

                        new_uncovered = total_space - np.sum(new_covered)
                        new_selected = selected_count - 1
                        new_energy = compute_energy(new_uncovered, new_selected)
                        delta = new_energy - current_energy

                        if delta <= 0 or random.random() < np.exp(-delta / T):
                            covered = new_covered
                            uncovered_count = new_uncovered
                            selected_count = new_selected
                            current_energy = new_energy

                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_selection = selection.copy()
                        else:
                            selection[w] = True

            else:
                # Swap: remove one selected, add one unselected
                selected_indices = np.where(selection)[0]
                unselected_indices = np.where(~selection)[0]
                if len(selected_indices) > 0 and len(unselected_indices) > 0:
                    w_remove = selected_indices[random.randint(0, len(selected_indices) - 1)]
                    w_add = unselected_indices[random.randint(0, len(unselected_indices) - 1)]

                    selection[w_remove] = False
                    selection[w_add] = True

                    new_covered = np.zeros(total_space, dtype=np.uint8)
                    for idx in range(total_space):
                        if selection[idx]:
                            new_covered[cover_matrix[idx]] = 1

                    new_uncovered = total_space - np.sum(new_covered)
                    new_selected = selected_count  # Same count after swap
                    new_energy = compute_energy(new_uncovered, new_selected)
                    delta = new_energy - current_energy

                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        covered = new_covered
                        uncovered_count = new_uncovered
                        selected_count = new_selected
                        current_energy = new_energy

                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_selection = selection.copy()
                    else:
                        selection[w_remove] = True
                        selection[w_add] = False

            T *= cooling_rate
            step += 1

        return best_selection, best_energy

    # Greedy set cover for initial solution
    def run_greedy():
        covered = np.zeros(total_space, dtype=bool)
        selected = []
        remaining = total_space

        while remaining > 0 and time.time() < deadline - 20:
            best_gain = -1
            best_word = -1
            candidates = []

            for w in range(total_space):
                gain = np.sum(cover_matrix[w] & ~covered)
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    candidates = [w]
                elif gain == best_gain:
                    candidates.append(w)

            if not candidates:
                break

            best_word = random.choice(candidates)
            selected.append(best_word)
            covered[cover_matrix[best_word]] = True
            remaining = total_space - np.sum(covered)

        return selected

    # Phase 1: Get a valid covering via greedy
    if time.time() < deadline - 30:
        greedy_sol = run_greedy()
        best_size = len(greedy_sol)
        best_selected = greedy_sol[:]

        # Polish
        polished = polish(best_selected[:], deadline - 25)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
    else:
        best_size = total_space
        best_selected = list(range(total_space))

    # Phase 2: Simulated Annealing from the greedy solution
    if time.time() < deadline - 10:
        sa_time_limit = deadline - 5
        best_sa_selection, best_sa_energy = run_simulated_annealing(best_selected, sa_time_limit)

        # Extract selected words
        sa_selected = list(np.where(best_sa_selection)[0])

        # Verify it's a valid cover
        covered = np.zeros(total_space, dtype=bool)
        for w in sa_selected:
            covered[cover_matrix[w]] = True

        if np.all(covered):
            # Polish
            polished = polish(sa_selected[:], deadline - 3)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]
        else:
            # SA didn't find a valid cover, stick with greedy
            pass

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 1)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Convert to list of tuples
    best_words = [int_to_tuple(v) for v in best_selected]

    return best_words
