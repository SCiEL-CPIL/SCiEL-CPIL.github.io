import time
import random
import numpy as np

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
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            if mono[ap_idx]:
                delta -= 1
            else:
                if offset == 0:
                    if color[p1] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 1:
                    if color[p0] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 2:
                    if color[p0] == new_val and color[p1] == new_val and color[p3] == new_val:
                        delta += 1
                else:
                    if color[p0] == new_val and color[p1] == new_val and color[p2] == new_val:
                        delta += 1
                        
        return delta

    def apply_flip(color, mono, p):
        """
        Flip position p in color, update mono status for affected APs.
        Returns the delta in count.
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            was_mono = mono[ap_idx]
            
            if offset == 0:
                is_mono = (color[p1] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 1:
                is_mono = (color[p0] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 2:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p3] == new_val)
            else:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p2] == new_val)
                
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

    def local_search_sa(color, mono, count, max_iter=5000, allow_two_flips=False):
        """
        Local search with Simulated Annealing.
        """
        T = 1.0
        cooling = 0.99
        min_T = 1e-6
        
        for _ in range(max_iter):
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
                    
            # Check two-flips if allowed and no single flip found
            if allow_two_flips and best_pos == -1:
                # Sample a few pairs
                for _ in range(20):
                    p1 = random.randint(0, n - 1)
                    p2 = random.randint(0, n - 1)
                    if p1 != p2:
                        d = delta_two_flips(color, mono, p1, p2)
                        if d < best_delta:
                            best_delta = d
                            best_pos = p1
                            best_pos2 = p2
                            # Mark that we have a two-flip move
                            # We need a way to distinguish. Let's use a flag.
                            # Actually, let's just store the move type.
                            pass # This logic is getting messy. Let's simplify.
            
            # Simplified: Just do single flips in SA, and use two-flips for perturbation in ILS.
            # This is safer and faster.
            
            if best_pos != -1:
                delta = apply_flip(color, mono, best_pos)
                count += delta
            else:
                if T < min_T:
                    break
                
                p = random.randint(0, n - 1)
                d = delta_flip(color, mono, p)
                
                if d < 0:
                    delta = apply_flip(color, mono, p)
                    count += delta
                else:
                    if random.random() < np.exp(-d / T):
                        delta = apply_flip(color, mono, p)
                        count += delta
                        
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
        
    random.seed(42)
    
    best_color = None
    best_score = float('inf')
    
    initializers = [
        thue_morse,
        random_init,
        rademacher_init,
        rademacher_4,
        lambda: [1 - x for x in thue_morse(n)],
    ]
    
    # Run ILS/SA on each
    for init_func in initializers:
        if time.time() > deadline - 15:
            break
            
        color = init_func(n)
        mono, count = init_state(color)
        
        # Run local search with SA
        color, mono, count = local_search_sa(color, mono, count, max_iter=3000)
        
        if count < best_score:
            best_score = count
            best_color = color[:]
            
        # Perform Iterated Local Search from this local optimum
        base_color = color[:]
        base_mono = mono[:]
        base_count = count
        
        # Multiple iterations of perturbation
        # Use two-flips for perturbation to escape local minima more effectively
        for _ in range(15):
            if time.time() > deadline - 5:
                break
                
            perturbed = base_color[:]
            # Perturb with 2-3 random two-flips
            num_two_flips = random.randint(2, 5)
            for _ in range(num_two_flips):
                p1 = random.randint(0, n - 1)
                p2 = random.randint(0, n - 1)
                if p1 != p2:
                    p_mono, p_count = init_state(perturbed)
                    # Just flip directly, don't need delta for perturbation
                    perturbed[p1] = 1 - perturbed[p1]
                    perturbed[p2] = 1 - perturbed[p2]
            
            p_mono, p_count = init_state(perturbed)
            
            # Local search on perturbed
            p_color, p_mono, p_count = local_search_sa(perturbed, p_mono, p_count, max_iter=1500)
            
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

    # Final polish: Run a longer local search on the best solution found
    if best_color is not None:
        final_mono, final_count = init_state(best_color)
        # Use a slightly higher T for more exploration? No, greedy is fine for polish.
        # Just run more iterations.
        best_color, final_mono, final_count = local_search_sa(best_color, final_mono, final_count, max_iter=10000)
        if final_count < best_score:
            best_score = final_count
            best_color = best_color[:]

    return best_color
