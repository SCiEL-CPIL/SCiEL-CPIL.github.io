"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools


def _precompute_neighbors(q, n, radius=1):
    """Precompute neighbor indices for all words in {0..q-1}^n within Hamming distance radius."""
    total = q ** n
    if radius == 1:
        num_neighbors = 1 + n * (q - 1)
        neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        for w in range(total):
            digits = np.array([w // q_powers[i] % q for i in range(n)], dtype=np.int64)
            neighbors[w, 0] = w
            idx = 1
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    neighbors[w, idx] = int(np.dot(digits, q_powers))
                    idx += 1
                digits[pos] = orig
        return neighbors
    else:
        raise NotImplementedError("Only radius 1 supported")


def _tetracode_initial(q, n):
    """Generate tetracode-based initial solution."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _word_to_idx(w, q_powers):
    return int(np.dot(np.array(w, dtype=np.int64), q_powers))


def _idx_to_word(idx, q, n):
    result = []
    for i in range(n - 1, -1, -1):
        result.append(idx % q)
        idx //= q
    result.reverse()
    return result


def _greedy_cover_fast(q, n, radius, time_limit=20.0):
    """Fast greedy: repeatedly pick candidate that covers most uncovered words."""
    start = time.time()
    total = q ** n
    neighbors = _precompute_neighbors(q, n, radius)
    num_neighbors = neighbors.shape[1]
    
    covered = np.zeros(total, dtype=np.int8)
    selected = []
    
    # Precompute: for each word, which codeword centers cover it (inverse adjacency)
    # center_covers[word] = list of centers c such that word in ball(c)
    # This is the transpose of neighbors
    
    while not covered.all():
        if time.time() - start > time_limit:
            break
        
        # Find uncovered words
        uncovered_mask = 1 - covered
        n_uncovered = uncovered_mask.sum()
        if n_uncovered == 0:
            break
        
        # For each candidate center, count how many uncovered words it covers
        # coverage[c] = covered[neighbors[c]] dot uncovered_mask... 
        # Actually: sum over j in neighbors[c] of uncovered_mask[j]
        # = neighbors @ uncovered_mask (vectorized)
        coverage = neighbors @ uncovered_mask
        best_c = int(np.argmax(coverage))
        
        if coverage[best_c] == 0:
            # No progress possible
            break
        
        selected.append(best_c)
        covered[neighbors[best_c]] = 1
    
    # Convert to words
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    return [_idx_to_word(c, q, n) for c in selected]


def _simulated_annealing_cover(q, n, radius, initial_codewords, time_limit=100.0):
    """
    Hypergraph vertex cover via simulated annealing on dual variables.
    
    Vertices: the 6561 words in {0,1,2}^n
    Hyperedges: 17-word spheres centered at each potential codeword
    Goal: select minimum hyperedges (codewords) to cover all vertices
    
    SA approach:
    - State: set of selected codewords (hyperedges)
    - Energy: number of uncovered vertices (should be 0 for valid solution)
      + penalty * (number of selected codewords) to minimize size
    - Moves: add a random hyperedge, remove a random hyperedge, swap
    - Accept if energy decreases or with probability exp(-delta_E / T)
    """
    start = time.time()
    total = q ** n
    neighbors = _precompute_neighbors(q, n, radius)
    num_neighbors = neighbors.shape[1]
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Convert initial codewords to indices
    initial_indices = [_word_to_idx(w, q_powers) for w in initial_codewords]
    
    # count[word] = number of selected centers whose ball contains word
    count = np.zeros(total, dtype=np.int32)
    for ci in initial_indices:
        count[neighbors[ci]] += 1
    
    # Track which centers are selected
    selected_set = set(initial_indices)
    selected_list = list(initial_indices)
    
    best_size = len(selected_list)
    best_set = set(selected_list)
    
    # Temperature schedule
    T0 = 5.0
    T_min = 0.01
    cooling = 0.995
    
    def compute_uncovered():
        return int((count == 0).sum())
    
    def energy():
        """Energy = uncovered_count + lambda * size"""
        return compute_uncovered() + 0.1 * len(selected_list)
    
    current_energy = energy()
    
    # Phase 1: Greedy removal of redundant codewords
    def is_redundant(ci):
        return count[neighbors[ci]].min() > 1
    
    changed = True
    while changed and time.time() - start < time_limit * 0.2:
        changed = False
        for i in range(len(selected_list) - 1, -1, -1):
            ci = selected_list[i]
            if is_redundant(ci):
                count[neighbors[ci]] -= 1
                selected_set.discard(ci)
                selected_list.pop(i)
                changed = True
                if len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_set = set(selected_list)
    
    # Phase 2: Simulated annealing with add/remove/swap moves
    T = T0
    all_centers = np.arange(total)
    rng = random.Random(42)
    
    sa_steps = 0
    max_sa_steps = 50000
    
    while time.time() - start < time_limit and sa_steps < max_sa_steps:
        sa_steps += 1
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4 and len(selected_list) > 0:
            # Remove a random selected center
            i = rng.randint(0, len(selected_list) - 1)
            ci = selected_list[i]
            
            # Check if removal would leave uncovered words
            if count[neighbors[ci]].min() > 1:
                # Safe to remove
                count[neighbors[ci]] -= 1
                selected_set.discard(ci)
                selected_list.pop(i)
                new_energy = energy()
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if len(selected_list) < best_size:
                        best_size = len(selected_list)
                        best_set = set(selected_list)
                else:
                    # Revert
                    selected_list.insert(i, ci)
                    selected_set.add(ci)
                    count[neighbors[ci]] += 1
                    
            elif rng.random() < np.exp(-1.0 / max(T, T_min)):
                # Accept removal even if it creates uncovered words (explore)
                count[neighbors[ci]] -= 1
                selected_set.discard(ci)
                selected_list.pop(i)
                new_energy = energy()
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if len(selected_list) < best_size:
                        best_size = len(selected_list)
                        best_set = set(selected_list)
                else:
                    # Revert
                    selected_list.insert(i, ci)
                    selected_set.add(ci)
                    count[neighbors[ci]] += 1
                    
        elif move_type < 0.7:
            # Add a random center
            cj = int(rng.choice(all_centers))
            if cj in selected_set:
                continue
            
            selected_list.append(cj)
            selected_set.add(cj)
            count[neighbors[cj]] += 1
            new_energy = energy()
            
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
                current_energy = new_energy
                # Try to remove redundant
                for j in range(len(selected_list) - 1, -1, -1):
                    if is_redundant(selected_list[j]):
                        cj2 = selected_list[j]
                        count[neighbors[cj2]] -= 1
                        selected_set.discard(cj2)
                        selected_list.pop(j)
                        new_energy = energy()
                        current_energy = new_energy
                        if len(selected_list) < best_size:
                            best_size = len(selected_list)
                            best_set = set(selected_list)
            else:
                # Revert
                count[neighbors[cj]] -= 1
                selected_set.discard(cj)
                selected_list.pop()
                
        else:
            # Swap: remove one, add one
            if len(selected_list) == 0:
                continue
            i = rng.randint(0, len(selected_list) - 1)
            ci = selected_list[i]
            cj = int(rng.choice(all_centers))
            if cj == ci or cj in selected_set:
                continue
            
            # Apply swap
            count[neighbors[ci]] -= 1
            count[neighbors[cj]] += 1
            selected_list[i] = cj
            selected_set.discard(ci)
            selected_set.add(cj)
            
            new_energy = energy()
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
                current_energy = new_energy
                # Try to remove redundant
                for j in range(len(selected_list) - 1, -1, -1):
                    if is_redundant(selected_list[j]):
                        cj2 = selected_list[j]
                        count[neighbors[cj2]] -= 1
                        selected_set.discard(cj2)
                        selected_list.pop(j)
                        new_energy = energy()
                        current_energy = new_energy
                        if len(selected_list) < best_size:
                            best_size = len(selected_list)
                            best_set = set(selected_list)
            else:
                # Revert
                count[neighbors[cj]] -= 1
                count[neighbors[ci]] += 1
                selected_list[i] = ci
                selected_set.discard(cj)
                selected_set.add(ci)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic reset if stuck
        if sa_steps % 5000 == 0 and len(selected_list) > best_size:
            # Reset to best
            count[:] = 0
            for ci in best_set:
                count[neighbors[ci]] += 1
            selected_list = list(best_set)
            selected_set = set(best_set)
            current_energy = energy()
            T = max(T, 1.0)
    
    # Final cleanup: remove all redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(selected_list) - 1, -1, -1):
            ci = selected_list[i]
            if is_redundant(ci):
                count[neighbors[ci]] -= 1
                selected_set.discard(ci)
                selected_list.pop(i)
                changed = True
    
    # Use best found
    if len(selected_list) > best_size:
        selected_list = list(best_set)
    
    # Verify coverage
    final_count = np.zeros(total, dtype=np.int32)
    for ci in selected_list:
        final_count[neighbors[ci]] += 1
    if final_count.min() < 1:
        # Invalid, fall back to best
        selected_list = list(best_set)
    
    return [_idx_to_word(ci, q, n) for ci in selected_list]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0  # Leave 15s margin for safety
    
    # Start with tetracode construction
    initial = _tetracode_initial(q, n)
    
    if len(initial) <= 486:
        return initial
    
    # Simulated annealing local search
    time_left = deadline - time.time()
    if time_left > 10:
        result = _simulated_annealing_cover(q, n, radius, initial, time_limit=time_left - 5)
        
        # Verify validity
        total = q ** n
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        neighbors = _precompute_neighbors(q, n, radius)
        count = np.zeros(total, dtype=np.int32)
        for w in result:
            ci = _word_to_idx(w, q_powers)
            count[neighbors[ci]] += 1
        
        if count.min() >= 1:
            return result
    
    # Fallback: return tetracode
    return initial
# EVOLVE-BLOCK-END
