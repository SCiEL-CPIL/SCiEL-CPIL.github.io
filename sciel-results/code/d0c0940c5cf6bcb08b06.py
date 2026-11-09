import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbor lists (indices within distance 1)
    neighbor_indices = [[] for _ in range(N)]
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                neigh.append(j)
        neighbor_indices[i] = neigh

    # Precompute coverage masks as Python integers (bitsets)
    masks = [0] * N
    for i in range(N):
        mask_val = 0
        for j in neighbor_indices[i]:
            mask_val |= (1 << j)
        masks[i] = mask_val

    full_mask = (1 << N) - 1

    def get_initial_tetracode():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = tuple(h + list(tail))
                init_indices.append(word_to_idx[w])
        return set(init_indices)

    def compute_cov(code_set):
        cov = 0
        for idx in code_set:
            cov |= masks[idx]
        return cov

    def is_redundant(idx, coverage_counts):
        for j in neighbor_indices[idx]:
            if coverage_counts[j] == 1:
                return False
        return True

    def polish(code_set, coverage_counts, cov, deadline):
        """Aggressive redundant removal with multiple orderings."""
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Try multiple random orderings
            for _ in range(3):
                indices_list = list(code_set)
                random.shuffle(indices_list)
                for idx in indices_list:
                    if time.time() > deadline - 2.0:
                        return code_set, coverage_counts, cov
                    if is_redundant(idx, coverage_counts):
                        for j in neighbor_indices[idx]:
                            coverage_counts[j] -= 1
                        code_set.remove(idx)
                        changed = True
        return code_set, coverage_counts, cov

    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    best_code = get_initial_tetracode()
    best_size = len(best_code)
    best_cov = compute_cov(best_code)

    all_indices = list(range(N))

    # Multi-start simulated annealing
    num_starts = 5
    time_per_start = (deadline - start_time) / num_starts

    for start_num in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with tetracode, add some randomness for diversity
        current_code = set(get_initial_tetracode())
        coverage_counts = np.zeros(N, dtype=np.int32)
        for idx in current_code:
            for j in neighbor_indices[idx]:
                coverage_counts[j] += 1
        current_cov = compute_cov(current_code)

        # If not fully covered, add random words until covered
        while current_cov != full_mask and time.time() < deadline - 5.0:
            uncovered = [i for i in range(N) if not (current_cov >> i) & 1]
            if not uncovered:
                break
            # Pick word that covers most uncovered
            best_idx = -1
            best_count = 0
            for i in uncovered:
                count = 0
                for j in neighbor_indices[i]:
                    if not (current_cov >> j) & 1:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            for j in neighbor_indices[best_idx]:
                coverage_counts[j] += 1
            current_code.add(best_idx)
            current_cov |= masks[best_idx]

        if current_cov != full_mask:
            continue

        # Polish initial solution
        current_code, coverage_counts, current_cov = polish(
            current_code, coverage_counts, current_cov, deadline
        )

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)

        # Simulated annealing
        T = 3.0
        T_min = 0.01
        cooling = 0.9995
        local_deadline = min(start_time + time_per_start + 2.0, deadline - 5.0)

        while time.time() < local_deadline and T > T_min:
            move_type = random.random()
            accepted = False
            new_code = current_code
            new_cov = current_cov
            new_counts = coverage_counts

            if move_type < 0.4:
                # Remove a random codeword
                if not current_code:
                    continue
                to_remove = random.choice(list(current_code))
                
                newly_uncovered = 0
                for j in neighbor_indices[to_remove]:
                    if coverage_counts[j] == 1:
                        newly_uncovered += 1
                
                delta_size = -1
                delta_uncovered = newly_uncovered
                delta_cost = delta_size + 1000 * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    new_counts = coverage_counts.copy()
                    for j in neighbor_indices[to_remove]:
                        new_counts[j] -= 1
                    new_code = set(current_code)
                    new_code.remove(to_remove)
                    if newly_uncovered == 0:
                        new_cov = current_cov
                    else:
                        new_cov = compute_cov(new_code)
                    accepted = True

            elif move_type < 0.7:
                # Add a random codeword
                to_add = random.choice(all_indices)
                if to_add in current_code:
                    continue
                
                new_cov_trial = current_cov | masks[to_add]
                newly_covered = new_cov_trial.bit_count() - current_cov.bit_count()
                
                delta_size = 1
                delta_uncovered = -newly_covered
                delta_cost = delta_size + 1000 * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    new_counts = coverage_counts.copy()
                    for j in neighbor_indices[to_add]:
                        new_counts[j] += 1
                    new_code = set(current_code)
                    new_code.add(to_add)
                    new_cov = new_cov_trial
                    accepted = True

            else:
                # Swap: remove one, add another
                if not current_code:
                    continue
                to_remove = random.choice(list(current_code))
                to_add = random.choice(all_indices)
                if to_add == to_remove or to_add in current_code:
                    continue
                
                # Efficient swap evaluation using bitset ops
                # New cov = (current_cov without to_remove's exclusive coverage) | masks[to_add]
                # Approximate: just recompute if swap looks promising
                # Quick check: does to_add cover any word that to_remove uniquely covers?
                to_remove_neigh = set(neighbor_indices[to_remove])
                to_add_neigh = set(neighbor_indices[to_add])
                
                # Words uniquely covered by to_remove
                unique_to_remove = to_remove_neigh & (set(neighbor_indices[j]) - to_add_neigh for j in current_code if j != to_remove)
                # This is expensive, so just do a quick heuristic:
                # If to_add covers more uncovered words than to_remove uniquely covers, try it
                
                # For speed, just recompute for swaps (they're less frequent)
                temp_code = set(current_code)
                temp_code.remove(to_remove)
                temp_code.add(to_add)
                new_cov_trial = compute_cov(temp_code)
                
                delta_uncovered = (N - new_cov_trial.bit_count()) - (N - current_cov.bit_count())
                delta_cost = 1000 * delta_uncovered  # delta_size = 0
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    new_counts = coverage_counts.copy()
                    for j in neighbor_indices[to_remove]:
                        new_counts[j] -= 1
                    for j in neighbor_indices[to_add]:
                        new_counts[j] += 1
                    new_code = temp_code
                    new_cov = new_cov_trial
                    accepted = True

            if accepted:
                current_code = new_code
                coverage_counts = new_counts
                current_cov = new_cov

                if current_cov == full_mask:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                        if best_size <= 73:
                            break

            T *= cooling

            # Periodic polish
            if random.random() < 0.08:
                current_code, coverage_counts, current_cov = polish(
                    current_code, coverage_counts, current_cov, local_deadline
                )
                if current_cov == full_mask and len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)

    # Final validation and return
    final_cov = 0
    for idx in best_code:
        final_cov |= masks[idx]
    if final_cov != full_mask:
        best_code = get_initial_tetracode()

    return [list(words[i]) for i in best_code]
