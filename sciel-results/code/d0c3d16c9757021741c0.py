import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)

    # For each index, precompute the list of AP indices it belongs to
    # and for each such AP, which other 3 indices are in it
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_aps[idx].append((ap_idx, others))

    def count_mono(color):
        """Count monochromatic 4-APs."""
        count = 0
        for i1, i2, i3, i4 in aps:
            c = color[i1]
            if c == color[i2] == color[i3] == color[i4]:
                count += 1
        return count

    def get_delta(color, idx, new_val):
        """Compute change in mono count if we flip color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx, others in index_to_aps[idx]:
            o0, o1, o2 = others
            if color[o0] == color[o1] == color[o2]:
                c_others = color[o0]
                if old_val == c_others:
                    if new_val != c_others:
                        delta -= 1
                else:
                    if new_val == c_others:
                        delta += 1
        return delta

    def local_search(color, max_iter=100):
        """Hill climbing: flip bits if it improves the score."""
        current_score = count_mono(color)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            for idx in indices:
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
            iter_count += 1
        return current_score

    # LFSR-based generation
    def generate_lfsr(taps, initial_state):
        """Generate a 256-bit sequence using LFSR over GF(2).
        taps: list of bit positions for feedback (0-indexed, LSB=0)
        initial_state: 256-bit numpy array
        """
        state = initial_state.copy()
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            seq[i] = state[0]
            # Compute feedback
            fb = 0
            for t in taps:
                fb ^= state[t]
            # Shift right and insert feedback at MSB
            state = np.roll(state, -1)
            state[-1] = fb
        return seq

    # Find primitive polynomials of degree 8 (for 255-bit period)
    # x^8 + x^4 + x^3 + x^2 + 1 (taps at positions 8,4,3,2,0 -> use 7,3,2,1,0 in 0-indexed)
    # Actually for LFSR, we use the polynomial coefficients
    # Let's use known primitive polynomials
    # x^8+x^4+x^3+x^2+1: taps = [7,3,2,1,0] (using 0-indexed, LSB first)
    # x^8+x^5+x^3+x^2+1: taps = [7,4,2,1,0]
    # x^8+x^6+x^5+x^4+1: taps = [7,5,4,3,0]
    
    primitive_taps = [
        [7, 3, 2, 1, 0],
        [7, 4, 2, 1, 0],
        [7, 5, 4, 3, 0],
        [7, 4, 3, 1, 0],
        [7, 6, 5, 4, 0],
    ]

    def lfsr_init(taps, seed):
        """Initialize LFSR with random state."""
        rng = random.Random(seed)
        initial_state = np.array([rng.randint(0, 1) for _ in range(8)], dtype=np.int8)
        return generate_lfsr(taps, initial_state)

    # Thue-Morse sequence
    def thue_morse_init():
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            # Count number of 1s in binary representation of i
            seq[i] = bin(i).count('1') % 2
        return seq

    # De Bruijn-like patterns
    def de_bruijn_init():
        # Use a pattern based on bit-reversal
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            # Reverse bits of i (8 bits)
            rev = 0
            x = i
            for _ in range(8):
                rev = (rev << 1) | (x & 1)
                x >>= 1
            seq[i] = rev & 1
        return seq

    # Generate diverse initial configurations
    candidates = []
    
    # LFSR-based inits
    for taps in primitive_taps:
        for seed in range(20):
            candidates.append(lfsr_init(taps, seed))
    
    # Thue-Morse
    candidates.append(thue_morse_init())
    
    # De Bruijn
    candidates.append(de_bruijn_init())
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]:
        candidates.append(np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8))
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11, 13, 17]:
        candidates.append(np.array([(i % p) < p // 2 for i in range(N)]).astype(np.int8))
    
    # XOR patterns
    for p1 in [3, 4, 5, 7]:
        for p2 in [5, 7, 9, 11]:
            if p1 != p2:
                a = (np.arange(N) % p1) < p1 // 2
                b = (np.arange(N) % p2) < p2 // 2
                candidates.append((a ^ b).astype(np.int8))
    
    # Random inits
    for _ in range(30):
        candidates.append(np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8))
    
    # Evaluate and refine candidates
    scored = []
    for c in candidates:
        if time.time() > deadline - 40.0:
            break
        c = c.copy()
        sc = local_search(c, max_iter=30)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Genetic Algorithm
    rng = random.Random(42)
    
    pop_size = 32
    population = []
    for i in range(min(pop_size, len(scored))):
        population.append(scored[i][1].copy())
    while len(population) < pop_size:
        population.append(np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8))
    
    best_score = scored[0][0]
    best_color = scored[0][1].copy()
    
    mutation_rate = 0.03
    elite_count = 6
    
    while time.time() < deadline - 15.0:
        scored_pop = []
        for ind in population:
            sc = count_mono(ind)
            scored_pop.append((sc, ind))
        
        scored_pop.sort(key=lambda x: x[0])
        
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_color = scored_pop[0][1].copy()
        
        new_population = [ind.copy() for _, ind in scored_pop[:elite_count]]
        
        scores_list = [s for s, _ in scored_pop]
        
        while len(new_population) < pop_size:
            if time.time() > deadline - 15.0:
                break
            
            # Tournament selection
            cands = rng.sample(range(len(population)), 3)
            best_idx = min(cands, key=lambda i: scores_list[i])
            parent1 = population[best_idx].copy()
            
            cands = rng.sample(range(len(population)), 3)
            best_idx = min(cands, key=lambda i: scores_list[i])
            parent2 = population[best_idx].copy()
            
            # Uniform crossover
            child = np.array([
                parent1[i] if rng.random() < 0.5 else parent2[i] 
                for i in range(N)
            ], dtype=np.int8)
            
            # Mutation
            for i in range(N):
                if rng.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            local_search(child, max_iter=20)
            new_population.append(child)
        
        population = new_population
    
    # Final polish with multi-bit flips
    if time.time() < deadline - 2.0:
        current = best_color.copy()
        current_score = count_mono(current)
        
        T = 2.0
        cooling = 0.9995
        max_iter = 20000
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            r = rng.random()
            if r < 0.6:
                # 1-bit flip
                idx = rng.randint(0, N - 1)
                new_val = 1 - current[idx]
                delta = get_delta(current, idx, new_val)
                
                if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                    current[idx] = new_val
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = current.copy()
            elif r < 0.85:
                # 2-bit flip
                idxs = rng.sample(range(N), 2)
                idx1, idx2 = idxs
                v1 = 1 - current[idx1]
                v2 = 1 - current[idx2]
                
                delta1 = get_delta(current, idx1, v1)
                current[idx1] = v1
                delta2 = get_delta(current, idx2, v2)
                total_delta = delta1 + delta2
                
                if total_delta < 0 or (total_delta > 0 and rng.random() < np.exp(-total_delta / T)):
                    current[idx2] = v2
                    current_score += total_delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = current.copy()
                else:
                    current[idx1] = 1 - v1
            else:
                # 3-bit flip
                idxs = rng.sample(range(N), 3)
                idx1, idx2, idx3 = idxs
                v1 = 1 - current[idx1]
                v2 = 1 - current[idx2]
                v3 = 1 - current[idx3]
                
                delta1 = get_delta(current, idx1, v1)
                current[idx1] = v1
                delta2 = get_delta(current, idx2, v2)
                current[idx2] = v2
                delta3 = get_delta(current, idx3, v3)
                total_delta = delta1 + delta2 + delta3
                
                if total_delta < 0 or (total_delta > 0 and rng.random() < np.exp(-total_delta / T)):
                    current[idx3] = v3
                    current_score += total_delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = current.copy()
                else:
                    current[idx1] = 1 - v1
                    current[idx2] = 1 - v2
            
            T *= cooling
        
        # Final local search
        local_search(best_color, max_iter=100)
        final_sc = count_mono(best_color)
        if final_sc < best_score:
            best_score = final_sc
    
    return [int(x) for x in best_color]
