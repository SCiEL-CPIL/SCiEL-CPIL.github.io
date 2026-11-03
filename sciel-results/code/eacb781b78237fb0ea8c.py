import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover bits for each word
    # cover_bits[i] is an integer where bit j is 1 if word j is within radius of word i
    cover_bits = [0] * total_space
    
    # Build cover bits
    # For each word i, find all words j within distance <= radius
    # Since radius is small (1), we can generate neighbors directly
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i  # self
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Create neighbor
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        
        cover_bits[i] = mask
    
    total_mask = (1 << total_space) - 1
    
    def popcount(x):
        return bin(x).count('1')

    def greedy_cover():
        uncovered = total_mask
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline - 10.0:
                break
                
            best_idx = -1
            best_gain = -1
            
            # Find word with max coverage of uncovered set
            for i in range(total_space):
                if i in code_set:
                    continue
                gain = popcount(cover_bits[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
                
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered &= ~cover_bits[best_idx]
            
        return code, uncovered

    def algorithm_x_min_cover(cover_bits, total_space, time_limit=30.0):
        """
        Algorithm X variant for minimum set cover using bit-parallel operations.
        Returns a list of set indices (codeword indices) forming a minimum cover,
        or None if no solution found within time limit.
        
        Uses Dancing Links (DLX) with bit-parallel column tracking.
        """
        start = time.time()
        
        # For each element (word), store which sets (codewords) cover it
        # element_to_sets[e] = list of set indices that cover element e
        element_to_sets = [[] for _ in range(total_space)]
        for s in range(total_space):
            mask = cover_bits[s]
            m = mask
            while m:
                lsb = m & -m
                e = lsb.bit_length() - 1
                element_to_sets[e].append(s)
                m ^= lsb
        
        # State for DLX:
        # active_elements: set of elements still needing coverage
        # active_sets: set of sets still available
        # cover_count[e]: number of active sets covering element e
        active_elements = set(range(total_space))
        active_sets = set(range(total_space))
        cover_count = np.zeros(total_space, dtype=np.int32)
        for e in range(total_space):
            cover_count[e] = len(element_to_sets[e])
        
        best_solution = None
        best_size = total_space + 1
        
        def solve(active_elements, active_sets, cover_count, current_solution, depth, max_depth):
            nonlocal best_solution, best_size
            
            if time.time() - start > time_limit:
                return False
            
            if not active_elements:
                if len(current_solution) < best_size:
                    best_size = len(current_solution)
                    best_solution = current_solution[:]
                return True
            
            if depth >= max_depth or len(current_solution) >= best_size:
                return False
            
            # Choose element with fewest active covering sets (MRV heuristic)
            min_count = float('inf')
            min_element = None
            for e in active_elements:
                cnt = cover_count[e]
                if cnt < min_count:
                    min_count = cnt
                    min_element = e
                if cnt == 0:
                    return False  # Dead end
            
            if min_count == 0:
                return False
            
            # Get active sets covering min_element
            candidates = [s for s in element_to_sets[min_element] if s in active_sets]
            
            # Sort candidates by how many additional elements they cover (prefer larger)
            candidates.sort(key=lambda s: -popcount(cover_bits[s] & active_elements))
            
            # Limit branching for performance
            max_branches = min(len(candidates), 8)
            
            for s in candidates[:max_branches]:
                # Include set s
                new_solution = current_solution + [s]
                
                # Compute new state
                new_active_elements = active_elements - set(cover_bits[s].bit_length() and 
                    [e for e in range(total_space) if (cover_bits[s] >> e) & 1])
                
                # More efficient: compute elements covered by s
                covered_by_s = []
                m = cover_bits[s]
                while m:
                    lsb = m & -m
                    e = lsb.bit_length() - 1
                    covered_by_s.append(e)
                    m ^= lsb
                covered_set = set(covered_by_s)
                
                new_active_elements = active_elements - covered_set
                
                if not new_active_elements:
                    if len(new_solution) < best_size:
                        best_size = len(new_solution)
                        best_solution = new_solution[:]
                    continue
                
                # Update cover counts
                new_cover_count = cover_count.copy()
                for e in new_active_elements:
                    if s in element_to_sets[e]:
                        new_cover_count[e] -= 1
                
                # Check for dead ends
                dead_end = False
                for e in new_active_elements:
                    if new_cover_count[e] == 0:
                        dead_end = True
                        break
                if dead_end:
                    continue
                
                new_active_sets = active_sets - {s}
                
                # Recurse
                if solve(new_active_elements, new_active_sets, new_cover_count, new_solution, depth + 1, max_depth):
                    # Found better solution, continue to find even better
                    pass
            
            return best_solution is not None
        
        # Initial call with max_depth = greedy_size - 1 to find smaller solutions
        greedy_code, _ = greedy_cover()
        max_depth = len(greedy_code) - 1
        
        if max_depth > 0:
            solve(set(range(total_space)), set(range(total_space)), cover_count, [], 0, max_depth)
        
        return best_solution

    def local_search_reduce(code, uncovered):
        if not code:
            return code, uncovered
            
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in code:
            mask = cover_bits[idx]
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb
        
        code_set = set(code)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
                
            random.shuffle(code)
            n_cw = len(code)
            
            for i in range(n_cw):
                idx = code[i]
                mask = cover_bits[idx]
                
                # Check if removable
                is_removable = True
                m = mask
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        is_removable = False
                        break
                    m ^= lsb
                
                if is_removable:
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                    
                    code.pop(i)
                    code_set.discard(idx)
                    improved = True
                    break
                else:
                    # Try swap
                    critical_mask = 0
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        if cover_count[bit_idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                    
                    if critical_mask:
                        found_swap = False
                        for k in range(total_space):
                            if k in code_set:
                                continue
                            if (critical_mask & ~cover_bits[k]) == 0:
                                # Remove idx
                                m = mask
                                while m:
                                    lsb = m & -m
                                    bit_idx = lsb.bit_length() - 1
                                    cover_count[bit_idx] -= 1
                                    m ^= lsb
                                    
                                # Add k
                                new_mask = cover_bits[k]
                                m = new_mask
                                while m:
                                    lsb = m & -m
                                    bit_idx = lsb.bit_length() - 1
                                    cover_count[bit_idx] += 1
                                    m ^= lsb
                                    
                                code[i] = k
                                code_set.discard(idx)
                                code_set.add(k)
                                improved = True
                                found_swap = True
                                break
                        
                        if found_swap:
                            break
                            
        return code, uncovered

    def simulated_annealing(code, uncovered):
        if not code:
            return code, uncovered
            
        current_code = code[:]
        current_uncovered = uncovered
        
        # Compute cover_count
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            m = cover_bits[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb
                
        def get_energy(c_uncovered, c_code):
            return len(c_code) + 100 * popcount(c_uncovered)
            
        current_energy = get_energy(current_uncovered, current_code)
        best_code = current_code[:]
        best_uncovered = current_uncovered
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 1.0
        cooling_rate = 0.999
        
        T = T_start
        sa_start = time.time()
        
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue
                    
                # New uncovered = current_uncovered & ~cover_bits[new_idx]
                new_uncovered = current_uncovered & ~cover_bits[new_idx]
                new_code = current_code + [new_idx]
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            elif move_type < 0.6:
                # Remove
                if len(current_code) <= 1:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_pos]
                
                # Find words uniquely covered by idx
                # unique_mask = cover_bits[idx] & (cover_count == 1)
                unique_mask = 0
                m = cover_bits[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                    
                new_uncovered = current_uncovered | unique_mask
                new_code = current_code[:remove_pos] + current_code[remove_pos+1:]
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    m = cover_bits[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            else:
                # Swap
                if not current_code:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)
                
                if new_idx == old_idx:
                    T *= cooling_rate
                    continue
                    
                # 1. Remove old_idx
                # unique_mask for old_idx
                unique_old = 0
                m = cover_bits[old_idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_old |= lsb
                    m ^= lsb
                    
                temp_uncovered = current_uncovered | unique_old
                
                # 2. Add new_idx
                new_uncovered = temp_uncovered & ~cover_bits[new_idx]
                
                new_code = current_code[:]
                new_code[remove_pos] = new_idx
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    # Remove old
                    m = cover_bits[old_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                        
                    # Add new
                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            T *= cooling_rate
            
        # Return best if valid (uncovered == 0)
        if best_uncovered == 0:
            return best_code, best_uncovered
        else:
            # Fallback to current if valid
            if current_uncovered == 0:
                return current_code, current_uncovered
            else:
                return code, uncovered

    # Initial greedy construction
    best_code, best_uncovered = greedy_cover()
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Try Algorithm X to find a smaller exact cover
    if time.time() < deadline - 60.0:
        ax_result = algorithm_x_min_cover(cover_bits, total_space, time_limit=40.0)
        if ax_result is not None and len(ax_result) < len(best_code):
            best_code = ax_result
            best_uncovered = 0  # Algorithm X guarantees full coverage
    
    # Run restarts if time permits
    while time.time() < deadline - 20.0:
        cand_code, cand_uncovered = greedy_cover()
        cand_code, _ = local_search_reduce(cand_code, cand_uncovered)
        
        # Apply simulated annealing to further reduce
        cand_code, cand_uncovered = simulated_annealing(cand_code, cand_uncovered)
        
        # Only accept if valid and smaller
        if cand_uncovered == 0 and len(cand_code) < len(best_code):
            best_code = cand_code
            best_uncovered = cand_uncovered
    
    # Final polish
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    if time.time() < deadline - 5.0:
        best_code, best_uncovered = simulated_annealing(best_code, best_uncovered)
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
