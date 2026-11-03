<<<<<<< SEARCH
    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = set()
        selected = []
        selected_set = set()

        while time.time() < deadline - 8:
            # Check if all covered
            if len(covered) == total_space:
                break
                
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                # Gain = number of new points covered
                gain = 0
                for pos in neighbor_lists[w]:
                    if pos not in covered:
                        gain += 1
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            for pos in neighbor_lists[best_word]:
                covered.add(pos)

        return selected
=======
    # Precompute neighbor lists as numpy arrays for vectorized operations
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbor_lists]
    
    def run_greedy(randomized=False, alpha=1.0, time_budget=None):
        """Greedy set cover with optional randomization."""
        if time_budget is None:
            time_budget = deadline - 8
        covered_arr = np.zeros(total_space, dtype=np.int8)
        selected = []
        selected_set = set()

        while time.time() < time_budget:
            # Check if all covered
            if covered_arr.sum() == total_space:
                break
                
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                # Gain = number of new points covered
                nbrs = neighbor_arrays[w]
                gain = np.count_nonzero(covered_arr[nbrs] == 0)
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered_arr[neighbor_arrays[best_word]] = 1

        return selected
>>>>>>> REPLACE

