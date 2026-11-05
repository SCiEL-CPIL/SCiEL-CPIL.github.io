import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def get_triple_indices(n):
        """Return list of all triples (a,b,c) with a<b<c."""
        return list(combinations(range(n), 3))

    def build_structure(n):
        """Build K4 and K5 structures for fast violation checking."""
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4: 4 vertices -> 4 triples
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        # K5: 5 vertices -> 10 triples
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Reverse indices: triple_idx -> list of clique indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_count = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s):
        """Compute cost delta when flipping triple t from old_color to new_color."""
        delta = 0

        # Check K4s containing t
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            red_count = sum(1 for i in idxs if colors[i] == 0)

            if old_color == 0:
                if red_count == 4:
                    delta -= 1
            else:
                if red_count == 3:
                    delta += 1

        # Check K5s containing t
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            blue_count = sum(1 for i in idxs if colors[i] == 1)

            if old_color == 1:
                if blue_count == 10:
                    delta -= 1
            else:
                if blue_count == 9:
                    delta += 1

        return delta

    def generate_cyclic_coloring(n, diff_set_mask):
        """
        Generate a coloring based on a cyclic difference set.
        Vertices are 0..n-1.
        A triple (a,b,c) is Red (0) if all pairwise differences (mod n) are in the diff_set.
        Otherwise Blue (1).
        diff_set_mask is an integer bitmask where bit i is 1 if difference i is in the set.
        Note: We must ensure symmetry: if d is in set, n-d is in set.
        """
        triples = get_triple_indices(n)
        colors = np.ones(len(triples), dtype=np.int8) # Default Blue
        
        # Precompute membership for differences
        # diff_mask has bits 0..n-1. We care about 1..n-1.
        # Symmetry: we should check min(d, n-d) or just ensure the mask is symmetric.
        
        for idx, (a, b, c) in enumerate(triples):
            d1 = (b - a) % n
            d2 = (c - b) % n
            d3 = (c - a) % n
            
            # Check if all differences are in the "Red Set"
            # We define Red if the differences are "small" or in the set.
            # Let's say Red if d1, d2, d3 are all present in the mask.
            # To make it symmetric, we should probably use min(d, n-d) for the mask check?
            # Or just ensure the mask itself is symmetric.
            
            # Let's use the raw difference mod n.
            # If the construction is cyclic, the property should hold for any rotation.
            
            if (diff_set_mask & (1 << d1)) and (diff_set_mask & (1 << d2)) and (diff_set_mask & (1 << d3)):
                colors[idx] = 0
            else:
                colors[idx] = 1
        return colors

    def optimize_cyclic_structure(n, time_limit, seed=0):
        """
        Optimize the difference set mask for a cyclic construction.
        """
        np.random.seed(seed)
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        # Initialize mask
        # Start with a random symmetric mask
        # We only need to define bits 1 to n//2. The rest are mirrored.
        half = n // 2
        init_bits = np.random.randint(0, 2, half)
        
        def mask_from_bits(bits):
            m = 0
            for i in range(1, n):
                if i <= half:
                    b = bits[i-1]
                else:
                    b = bits[n-i-1]
                if b:
                    m |= (1 << i)
            return m

        current_mask = mask_from_bits(init_bits)
        current_colors = generate_cyclic_coloring(n, current_mask)
        current_cost = count_violations(current_colors, k4s, k5s)
        best_mask = current_mask
        best_cost = current_cost
        best_colors = current_colors.copy()
        
        if best_cost == 0:
            return best_colors, True

        # Local search on the mask bits
        temp = 1.0
        cool_rate = 0.999
        min_temp = 0.01
        
        step = 0
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < time_limit:
            step += 1
            
            # Flip a random bit in the symmetric mask
            # We pick a difference d in 1..n-1
            d = np.random.randint(1, n)
            
            old_bit = (current_mask >> d) & 1
            new_bit = 1 - old_bit
            
            # Update mask
            if old_bit:
                new_mask = current_mask & ~(1 << d)
            else:
                new_mask = current_mask | (1 << d)
                
            # We must maintain symmetry!
            # If we flip d, we must flip n-d.
            # If d == n-d (i.e., n is even and d=n/2), it's self-symmetric.
            
            # Let's just enforce symmetry in the mask generation or check both.
            # Easier: operate on the "half" representation.
            
            # Re-implement search on the half-bits for clarity and speed
            # This function is getting complex. Let's simplify the search strategy.
            
            break # Break out to use a simpler direct bit flip on the full mask with symmetry enforcement
            
        # Simpler approach: Random restarts of local search on the mask
        for restart in range(50):
            if time.time() > time_limit:
                break
                
            bits = np.random.randint(0, 2, half)
            current_mask = mask_from_bits(bits)
            current_colors = generate_cyclic_coloring(n, current_mask)
            current_cost = count_violations(current_colors, k4s, k5s)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = current_mask
                best_colors = current_colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Local search on this mask
            temp = 0.5
            for _ in range(500):
                if time.time() > time_limit:
                    break
                
                d = np.random.randint(1, n)
                # Flip d and n-d
                d2 = (n - d) % n
                if d2 == 0: d2 = n # handle n-d=0 case? No, d in 1..n-1, n-d in 1..n-1.
                
                # Check if d and d2 are distinct
                if d == d2:
                    # Self-symmetric
                    old_bit = (current_mask >> d) & 1
                    if old_bit:
                        new_mask = current_mask & ~(1 << d)
                    else:
                        new_mask = current_mask | (1 << d)
                else:
                    old_bit_d = (current_mask >> d) & 1
                    old_bit_d2 = (current_mask >> d2) & 1
                    # They should be the same due to symmetry, but let's force it
                    new_bit = 1 - old_bit_d
                    
                    new_mask = current_mask
                    if old_bit_d:
                        new_mask &= ~(1 << d)
                        new_mask &= ~(1 << d2)
                    else:
                        new_mask |= (1 << d)
                        new_mask |= (1 << d2)
                
                new_colors = generate_cyclic_coloring(n, new_mask)
                new_cost = count_violations(new_colors, k4s, k5s)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_mask = new_mask
                    current_colors = new_colors
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_mask = current_mask
                        best_colors = current_colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif temp > min_temp:
                    if np.random.rand() < np.exp(-(new_cost - current_cost) / temp):
                        current_cost = new_cost
                        current_mask = new_mask
                        current_colors = new_colors
                
                temp *= cool_rate
                
        return best_colors, (best_cost == 0)

    def local_search_refine(n, time_limit, initial_colors, seed=0):
        """
        Refine a specific coloring using local search.
        """
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        np.random.seed(seed)
        
        colors = initial_colors.copy()
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        
        if best_cost == 0:
            return best_colors, True
            
        temp = 1.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        step = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < time_limit:
            step += 1
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            dc = delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + dc
            
            if dc <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-dc / temp)
                else:
                    accept = False
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            temp *= cool_rate
            
            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.5, temp * 1.5)
                no_improve = 0
            
            if no_improve > max_no_improve * 4:
                colors = best_colors.copy()
                current_cost = best_cost
                temp = max(0.8, temp)
                no_improve = 0

        return best_colors, (best_cost == 0)

    def verify(n, colors):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    target_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        if n >= 30:
            time_budget = min(25.0, remaining * 0.6)
        elif n >= 25:
            time_budget = min(18.0, remaining * 0.5)
        elif n >= 20:
            time_budget = min(12.0, remaining * 0.4)
        else:
            time_budget = min(8.0, remaining * 0.3)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False

        # Strategy 1: Cyclic Structure Optimization
        # This is the "Structural Change"
        try:
            colors_cyclic, is_valid = optimize_cyclic_structure(n, time_limit * 0.5, seed=42)
            if is_valid:
                if verify(n, colors_cyclic.tolist()):
                    final_n = n
                    final_colors = colors_cyclic.tolist()
                    found = True
                    break
            else:
                # Use the best cyclic coloring as a seed for local search
                remaining_time = time_limit - time.time()
                if remaining_time > 2:
                    colors_refined, is_valid = local_search_refine(n, remaining_time, colors_cyclic, seed=123)
                    if is_valid:
                        if verify(n, colors_refined.tolist()):
                            final_n = n
                            final_colors = colors_refined.tolist()
                            found = True
                            break
        except Exception:
            pass

        # Strategy 2: Standard Local Search (Fallback / Diversification)
        if not found:
            remaining_time = time_limit - time.time()
            if remaining_time > 2:
                # Random start
                np.random.seed(99)
                triples = get_triple_indices(n)
                colors_rand = np.array([np.random.randint(0, 2) for _ in range(len(triples))], dtype=np.int8)
                colors_refined, is_valid = local_search_refine(n, remaining_time, colors_rand, seed=55)
                if is_valid:
                    if verify(n, colors_refined.tolist()):
                        final_n = n
                        final_colors = colors_refined.tolist()
                        found = True
                        break

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            np.random.seed(42)
            for _ in range(1000):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            else:
                final_n = 1
                final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
