import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s limit to be safe within 90s wall-clock
    
    n = 128
    target_start = 128
    target_end = 256
    target_indices = np.arange(target_start, target_end + 1)
    num_targets = target_end - target_start + 1  # 129 targets
    
    # Precompute FFT setup for O(N log N) multiplicity updates
    # We'll use FFT to compute convolution in O(N log N)
    # v is length 129, we need conv up to index 256, so we need FFT size >= 257
    fft_size = 512  # Next power of 2 >= 257
    
    def compute_multiplicities(v):
        """
        Compute unordered representation multiplicities for target range [128, 256]
        using FFT. Returns array of shape (129,) with multiplicity for each target.
        v: indicator vector of length 129 (0/1 values)
        """
        # Pad v to fft_size
        v_padded = np.zeros(fft_size, dtype=np.float64)
        v_padded[:129] = v
        
        # FFT
        V = np.fft.fft(v_padded)
        
        # Convolution in frequency domain
        conv_freq = V * V
        
        # Inverse FFT
        conv = np.fft.ifft(conv_freq).real
        
        # Extract target range [128, 256]
        ordered = conv[target_start:target_end + 1]
        
        # Calculate diagonal contributions
        # For sum t, if t is even and t/2 is in A, diagonal contribution is 1
        diag = np.zeros(num_targets, dtype=np.float64)
        even_mask = (target_indices % 2 == 0)
        if np.any(even_mask):
            even_indices = target_indices[even_mask]
            mid_indices = even_indices // 2
            diag[even_mask] = v[mid_indices]
            
        unordered = (ordered + diag) / 2.0
        
        return unordered

    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered).
        """
        if not A_set:
            return 1000, num_targets
        
        v = np.zeros(129, dtype=np.float64)
        for x in A_set:
            v[x] = 1.0
        
        unordered = compute_multiplicities(v)
        
        max_mult = int(np.ceil(np.max(unordered)))
        num_uncovered = int(np.sum(unordered < 0.5))
        
        return max_mult, num_uncovered

    def cost(A_set):
        m, u = evaluate(A_set)
        return u * 10000 + m

    # 1. Smart Initialization with spectral-guided search
    best_A = set()
    best_cost = float('inf')
    
    # Gradient-descent-like local search using FFT-based multiplicity computation
    # Key insight: flipping bit x changes multiplicities in a structured way
    # We can compute the effect of adding/removing x efficiently
    
    def compute_flip_effect(v, x, add=True):
        """
        Compute how multiplicities change when we flip bit x.
        If add=True, we're adding x; if add=False, we're removing x.
        Returns delta_unordered: change in unordered multiplicities for target range.
        """
        # When we add/remove x, the convolution changes by:
        # delta_conv = v * e_x + e_x * v (for add) or negative (for remove)
        # where e_x is unit vector at position x
        # In frequency domain: delta_V = ±e_x
        # delta_conv_freq = V * delta_V + delta_V * V = 2*V*delta_V (for real signals)
        
        # But we can compute directly: the change in ordered conv at position t
        # is v[t-x] + v[x] - v[x]*v[t-x] for add (when t-x is valid)
        # Actually simpler: delta_ordered[t] = v[t-x] + v[x] - 2*v[x]*v[t-x] if adding
        # Wait, let me think more carefully.
        
        # Original ordered conv: sum over a,b: v[a]*v[b]*[a+b=t]
        # After adding x (v[x] goes from 0 to 1):
        # New conv = old conv + v[x]*(v[t-x]) + v[t-x]*(v[x]) - v[x]*v[t-x]*v[x]
        # But v[x] was 0, so new = old + v[t-x] + v[t-x] - 0 = old + 2*v[t-x]
        # Wait, that's not right either. Let me redo.
        
        # Old: sum_{a+b=t} v[a]*v[b]
        # New: sum_{a+b=t} v'[a]*v'[b] where v'[x]=1, v'[a]=v[a] for a!=x
        # = sum_{a+b=t, a!=x, b!=x} v[a]*v[b] + v'[x]*v'[t-x] + v'[t-x]*v'[x] - v'[x]*v'[x] if x+x=t
        # This is getting complex. Let's just use FFT for the flip effect.
        
        # Actually, for efficiency, let's compute the full multiplicity array
        # but we can cache the FFT of the current v.
        
        # For now, let's just compute directly - it's O(129) per flip which is fine
        delta_ordered = np.zeros(num_targets)
        sign = 1.0 if add else -1.0
        
        for i, t in enumerate(target_indices):
            a = x
            b = t - x
            if 0 <= b <= 128:
                # The change in ordered conv when flipping x:
                # If adding: we add v[b] (from pair x,b) and v[a] (from pair b,x) 
                # but need to be careful about double counting and self-pairs
                # Ordered conv change: if we set v[x] from 0 to 1:
                # delta = v[b] + v[a] - v[a]*v[b] where a=x, b=t-x
                # But v[a] was 0 before, so delta = v[b] + 0 - 0 = v[b]
                # Wait, I'm overcomplicating. Let me just recompute.
                
                # For ordered convolution, the contribution of pairs involving x:
                # (x, t-x) and (t-x, x) if t-x != x
                # (x, x) if 2*x == t
                # 
                # When adding x (v[x]: 0->1):
                # delta_ordered[t] = v[t-x] + (v[t-x] if t-x != x else 0) 
                #   = v[t-x] + v[t-x] - v[x]*v[t-x] (but v[x] was 0)
                #   = 2*v[t-x] if t-x != x, or 1 if 2*x==t
                # Actually: the new terms are v'[x]*v'[t-x] and v'[t-x]*v'[x]
                # = 1*v[t-x] + v[t-x]*1 = 2*v[t-x] (if t-x != x)
                # If t-x == x, then it's just v'[x]*v'[x] = 1
                # And we remove the old terms which were 0 since v[x]=0
                # So delta = 2*v[t-x] if t-x != x, else 1
                # But wait, v[t-x] might also be 0
                # 
                # Hmm, let me just do it cleanly:
                if b == x:
                    # Self-pair: old contribution was v[x]*v[x] = 0, new is 1*1 = 1
                    delta_ordered[i] = sign * 1.0
                else:
                    # Two ordered pairs: (x,b) and (b,x)
                    # Old: v[x]*v[b] + v[b]*v[x] = 0
                    # New: 1*v[b] + v[b]*1 = 2*v[b]
                    delta_ordered[i] = sign * 2.0 * v[b]
            # If b out of range, no contribution from pairs involving x for this t
        
        # Now convert to unordered change
        # Unordered = (ordered + diag) / 2
        # The diagonal also changes: if t is even and t/2 == x, diag changes by sign
        delta_diag = np.zeros(num_targets)
        even_mask = (target_indices % 2 == 0)
        if np.any(even_mask):
            even_indices = target_indices[even_mask]
            mid_indices = even_indices // 2
            # If x is the midpoint for some target, diag changes
            for i in range(num_targets):
                if even_mask[i] and mid_indices[i] == x:
                    delta_diag[i] = sign
        
        delta_unordered = (delta_ordered + delta_diag) / 2.0
        return delta_unordered
    
    # Use spectral initialization: start with a structured set that has good coverage
    # Then use gradient-like search to minimize variance
    
    # First, get a valid covering set using greedy
    def greedy_cover():
        """Build a set that covers all targets using greedy approach."""
        A = {128}  # 128+128=256, needed for max target
        v = np.zeros(129, dtype=np.float64)
        v[128] = 1.0
        
        # Compute current coverage
        unordered = compute_multiplicities(v)
        uncovered = np.where(unordered < 0.5)[0]
        
        max_iters = 50
        for _ in range(max_iters):
            if len(uncovered) == 0 or time.time() > deadline - 15:
                break
            
            # Find the element that covers the most uncovered targets
            best_x = None
            best_count = 0
            for x in range(129):
                if v[x] > 0.5:
                    continue
                # How many uncovered targets would x help cover?
                # x helps cover target t if x + y = t for some y in A
                # i.e., t - x is in A
                count = 0
                for i in uncovered:
                    t = target_indices[i]
                    y = t - x
                    if 0 <= y <= 128 and v[y] > 0.5:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_x = x
            
            if best_x is None or best_count == 0:
                break
            
            A.add(best_x)
            v[best_x] = 1.0
            unordered = compute_multiplicities(v)
            uncovered = np.where(unordered < 0.5)[0]
        
        return A
    
    # Initialize with greedy cover
    best_A = greedy_cover()
    best_cost = cost(best_A)
    
    # Spectral-guided local search
    # Objective: minimize max multiplicity while maintaining coverage
    # Use variance as a smooth surrogate: minimize sum of (mult - mean)^2
    
    def spectral_cost(v):
        """
        Compute a smooth cost function based on multiplicity variance.
        Lower variance means more uniform representation counts.
        """
        unordered = compute_multiplicities(v)
        
        # Coverage penalty
        uncovered = np.sum(unordered < 0.5)
        
        # Variance of multiplicities (smooth surrogate for max)
        mean_mult = np.mean(unordered)
        variance = np.mean((unordered - mean_mult) ** 2)
        
        # Max multiplicity
        max_mult = np.max(unordered)
        
        # Combined cost: prioritize coverage, then minimize max, then variance
        cost_val = uncovered * 10000 + max_mult * 10 + variance
        
        return cost_val, unordered, uncovered, max_mult

    # 2. Spectral gradient-descent local search
    current_A = set(best_A)
    v = np.zeros(129, dtype=np.float64)
    for x in current_A:
        v[x] = 1.0
    
    current_cost, current_unordered, current_uncovered, current_max = spectral_cost(v)
    best_global_A = set(current_A)
    best_global_c = current_cost
    
    T = 3.0
    T_min = 0.1
    cooling = 0.995
    
    steps = 0
    while time.time() < deadline - 3:
        steps += 1
        
        # Choose a flip: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.35 and len(current_A) > 1:
            # Remove an element
            x = random.choice(list(current_A))
            new_v = v.copy()
            new_v[x] = 0.0
            new_A = current_A - {x}
        elif move_type < 0.65:
            # Add an element
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_v = v.copy()
            new_v[x] = 1.0
            new_A = current_A | {x}
        else:
            # Swap
            if len(current_A) < 2:
                continue
            x = random.choice(list(current_A))
            y = random.randint(0, 128)
            if y == x:
                continue
            new_v = v.copy()
            new_v[x] = 0.0
            new_v[y] = 1.0
            new_A = (current_A - {x}) | {y}
        
        if not new_A:
            continue
        
        new_cost, new_unordered, new_uncovered, new_max = spectral_cost(new_v)
        delta = new_cost - current_cost
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            current_A = new_A
            v = new_v
            current_cost = new_cost
            current_unordered = new_unordered
            current_uncovered = new_uncovered
            current_max = new_max
            
            # Track best by actual cost (max mult with coverage)
            m, u = evaluate(current_A)
            actual_c = u * 10000 + m
            if actual_c < best_global_c:
                best_global_c = actual_c
                best_global_A = set(current_A)
        
        T = max(T * cooling, T_min)
        
    # 3. Final Deterministic Polish: try all single removals and swaps
    # Multiple passes until no improvement
    
    max_polish_passes = 5
    for _pass in range(max_polish_passes):
        if time.time() > deadline - 2:
            break
        
        improved = False
        m_best, u_best = evaluate(best_global_A)
        
        # Try removing each element
        for x in sorted(list(best_global_A)):
            if time.time() > deadline - 2:
                break
            if len(best_global_A) == 1:
                break
            
            test_A = best_global_A - {x}
            if not test_A:
                continue
            m_test, u_test = evaluate(test_A)
            
            if u_test == 0 and u_best == 0:
                if m_test < m_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break
            elif u_best > 0:
                if u_test < u_best or (u_test == u_best and m_test < m_best):
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break
        
        if not improved:
            # Try swaps
            for x in sorted(list(best_global_A)):
                if time.time() > deadline - 2:
                    break
                for y in range(129):
                    if y == x or y in best_global_A:
                        continue
                    test_A = (best_global_A - {x}) | {y}
                    m_test, u_test = evaluate(test_A)
                    
                    if u_test == 0 and u_best == 0:
                        if m_test < m_best:
                            best_global_A = test_A
                            improved = True
                            m_best, u_best = evaluate(best_global_A)
                            break
                    elif u_best > 0:
                        if u_test < u_best or (u_test == u_best and m_test < m_best):
                            best_global_A = test_A
                            improved = True
                            m_best, u_best = evaluate(best_global_A)
                            break
                if improved:
                    break
        
        if not improved:
            break

    # Ensure 128 is in the set
    if 128 not in best_global_A:
        best_global_A.add(128)

    # Final validation
    m_final, u_final = evaluate(best_global_A)
    if u_final > 0:
        # Coverage failed, fall back to greedy
        greedy_A = greedy_cover()
        if greedy_A:
            best_global_A = greedy_A

    return sorted(best_global_A)
