import time
import random
import numpy as np
import math

def construct():
    n = 256
    # Deadline: finish well before 150s kill. Target 118s.
    deadline = time.time() + 118

    # 1. Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # Map position to list of (ap_index, offset_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Precompute APs as numpy array for fast full counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono_numpy(color):
        """Fast count using NumPy."""
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    # Precompute for each position: list of (ap_idx, offset, other_positions_tuple)
    # This allows faster delta calculation
    pos_info = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_info[p0].append((idx, 0, (p1, p2, p3)))
        pos_info[p1].append((idx, 1, (p0, p2, p3)))
        pos_info[p2].append((idx, 2, (p0, p1, p3)))
        pos_info[p3].append((idx, 3, (p0, p1, p2)))
    
    def init_state(color):
        """Initialize mono status and count."""
        mono = [False] * num_aps
        count = 0
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                count += 1
        return mono, count

    def delta_flip(color, mono, p):
        """
        Calculate the change in mono count if we flip position p.
        Uses precomputed pos_info for speed.
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset, others in pos_info[p]:
            if mono[ap_idx]:
                delta -= 1
            else:
                # Check if all other positions match new_val
                if color[others[0]] == new_val and color[others[1]] == new_val and color[others[2]] == new_val:
                    delta += 1
                        
        return delta

    def apply_flip(color, mono, p):
        """
        Flip position p in color, update mono status for affected APs.
        Returns the delta in count.
        Uses precomputed pos_info for speed.
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset, others in pos_info[p]:
            was_mono = mono[ap_idx]
            is_mono = (color[others[0]] == new_val and color[others[1]] == new_val and color[others[2]] == new_val)
                
            mono[ap_idx] = is_mono
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        color[p] = new_val
        return delta

    def delta_two_flips(color, mono, p1, p2):
        """
        Calculate change in mono count if we flip both p1 and p2.
        """
        delta = 0
        new_val1 = 1 - color[p1]
        new_val2 = 1 - color[p2]
        
        # We need to consider all APs that contain p1 or p2.
        # An AP is affected if it contains p1, p2, or both.
        # If it contains both, the net effect depends on the other two.
        # If it contains only one, the effect is like a single flip but
        # the "new value" of the other position is its current value.
        
        # To avoid double counting or missing interactions, we can just
        # simulate the state of each affected AP.
        
        # Collect unique AP indices
        affected_aps = set()
        for idx, _ in pos_aps[p1]:
            affected_aps.add(idx)
        for idx, _ in pos_aps[p2]:
            affected_aps.add(idx)
            
        for ap_idx in affected_aps:
            p0, p1_idx, p2_idx, p3_idx = aps[ap_idx]
            
            # Determine the new values for the 4 positions
            # Start with current values
            vals = [color[p0], color[p1_idx], color[p2_idx], color[p3_idx]]
            
            # Apply flips
            if p1_idx == p0: vals[0] = new_val1
            elif p1_idx == p1_idx: vals[1] = new_val1
            elif p1_idx == p2_idx: vals[2] = new_val1
            elif p1_idx == p3_idx: vals[3] = new_val1
            
            if p2_idx == p0: vals[0] = new_val2
            elif p2_idx == p1_idx: vals[1] = new_val2
            elif p2_idx == p2_idx: vals[2] = new_val2
            elif p2_idx == p3_idx: vals[3] = new_val2
            
            is_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            
            if is_mono and not mono[ap_idx]:
                delta += 1
            elif not is_mono and mono[ap_idx]:
                delta -= 1
                
        return delta

    def apply_two_flips(color, mono, p1, p2):
        """
        Flip positions p1 and p2.
        """
        delta = 0
        new_val1 = 1 - color[p1]
        new_val2 = 1 - color[p2]
        
        affected_aps = set()
        for idx, _ in pos_aps[p1]:
            affected_aps.add(idx)
        for idx, _ in pos_aps[p2]:
            affected_aps.add(idx)
            
        for ap_idx in affected_aps:
            p0, p1_idx, p2_idx, p3_idx = aps[ap_idx]
            
            vals = [color[p0], color[p1_idx], color[p2_idx], color[p3_idx]]
            
            if p1_idx == p0: vals[0] = new_val1
            elif p1_idx == p1_idx: vals[1] = new_val1
            elif p1_idx == p2_idx: vals[2] = new_val1
            elif p1_idx == p3_idx: vals[3] = new_val1
            
            if p2_idx == p0: vals[0] = new_val2
            elif p2_idx == p1_idx: vals[1] = new_val2
            elif p2_idx == p2_idx: vals[2] = new_val2
            elif p2_idx == p3_idx: vals[3] = new_val2
            
            is_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            was_mono = mono[ap_idx]
            
            mono[ap_idx] = is_mono
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        color[p1] = new_val1
        color[p2] = new_val2
        return delta

    def local_search_sa(color, mono, count, max_iter=5000):
        """
        Local search with Simulated Annealing.
        Uses dynamic temperature and more aggressive exploration.
        """
        T = 2.0
        cooling = 0.985
        min_T = 1e-4
        no_improve = 0
        best_local = count
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Find best flip (greedy)
            best_delta = 0
            best_pos = -1
            
            for p in range(n):
                d = delta_flip(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            
            if best_pos != -1:
                delta = apply_flip(color, mono, best_pos)
                count += delta
                no_improve = 0
                if count < best_local:
                    best_local = count
            else:
                # No improving move: use SA to escape
                if T < min_T:
                    # Try a few random 2-flip moves as perturbation
                    found = False
                    for _ in range(10):
                        p1 = random.randint(0, n - 1)
                        p2 = random.randint(0, n - 1)
                        if p1 != p2:
                            d = delta_two_flips(color, mono, p1, p2)
                            if d < 0:
                                apply_two_flips(color, mono, p1, p2)
                                count += d
                                found = True
                                break
                    if not found:
                        break
                    
                    no_improve += 1
                    if no_improve > 5:
                        break
                else:
                    p = random.randint(0, n - 1)
                    d = delta_flip(color, mono, p)
                    
                    if d < 0:
                        delta = apply_flip(color, mono, p)
                        count += delta
                        no_improve = 0
                        if count < best_local:
                            best_local = count
                    else:
                        # Accept with probability based on temperature
                        if random.random() < math.exp(-d / T):
                            delta = apply_flip(color, mono, p)
                            count += delta
                        no_improve += 1
                    
                    T *= cooling
        
        return color, mono, count

    # Initializations
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]
        
    def rademacher_init(n):
        return [(i // 2) % 2 for i in range(n)]
        
    def rademacher_4(n):
        return [(i // 4) % 2 for i in range(n)]
        
    def walsh_hadamard_init(n):
        """Walsh-Hadamard based coloring."""
        result = []
        for i in range(n):
            # Use bit-reversal for more varied pattern
            rev = int(bin(i)[2:].zfill(8)[::-1], 2)
            result.append(bin(rev).count('1') % 2)
        return result
    
    random.seed(12345)
    
    best_color = None
    best_score = float('inf')
    
    initializers = [
        thue_morse,
        random_init,
        rademacher_init,
        rademacher_4,
        walsh_hadamard_init,
        lambda: [1 - x for x in thue_morse(n)],
    ]
    
    # Run ILS/SA on each
    for init_func in initializers:
        if time.time() > deadline - 20:
            break
            
        color = init_func(n)
        mono, count = init_state(color)
        
        # Run local search with SA
        color, mono, count = local_search_sa(color, mono, count, max_iter=2500)
        
        if count < best_score:
            best_score = count
            best_color = color[:]
            
        # Perform Iterated Local Search from this local optimum
        base_color = color[:]
        base_mono = mono[:]
        base_count = count
        
        # Multiple iterations of perturbation
        for _ in range(20):
            if time.time() > deadline - 8:
                break
                
            perturbed = base_color[:]
            # Perturb with random number of single flips (1-8)
            num_flips = random.randint(1, 8)
            for _ in range(num_flips):
                p = random.randint(0, n - 1)
                perturbed[p] = 1 - perturbed[p]
            
            p_mono, p_count = init_state(perturbed)
            
            # Local search on perturbed
            p_color, p_mono, p_count = local_search_sa(perturbed, p_mono, p_count, max_iter=1200)
            
            if p_count < best_score:
                best_score = p_count
                best_color = p_color[:]
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count
                
            if p_count < base_count:
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count

    # Final high-precision polish: exhaustive local search on best solution
    if best_color is not None:
        final_mono, final_count = init_state(best_color)
        
        # Phase 1: Greedy descent
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for p in range(n):
                d = delta_flip(best_color, final_mono, p)
                if d < 0:
                    delta = apply_flip(best_color, final_mono, p)
                    final_count += delta
                    improved = True
        
        # Phase 2: Try all 2-flip moves
        if time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for p1 in range(n):
                    if time.time() > deadline - 2:
                        break
                    for p2 in range(p1 + 1, n):
                        d = delta_two_flips(best_color, final_mono, p1, p2)
                        if d < 0:
                            delta = apply_two_flips(best_color, final_mono, p1, p2)
                            final_count += delta
                            improved = True
        
        if final_count < best_score:
            best_score = final_count
            best_color = best_color[:]

    return best_color
