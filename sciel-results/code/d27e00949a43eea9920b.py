import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    M = num_aps
    
    # Precompute adjacency: for each position, which APs and roles
    # This allows faster delta computation
    # For each position p, store list of (ap_idx, role)
    pos_ap_roles = [[] for _ in range(N)]
    for ap_idx in range(M):
        i0, i1, i2, i3 = aps_array[ap_idx]
        pos_ap_roles[i0].append((ap_idx, 0))
        pos_ap_roles[i1].append((ap_idx, 1))
        pos_ap_roles[i2].append((ap_idx, 2))
        pos_ap_roles[i3].append((ap_idx, 3))
    
    def count_mono(colors):
        """Count monochromatic 4-term APs."""
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        mono0 = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        mono1 = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        return int(np.sum(mono0) + np.sum(mono1))
    
    def compute_deltas(colors):
        """
        Compute the change in mono count if we flip each bit.
        Uses vectorized operations.
        """
        c = colors
        c_aps = c[aps_array]  # Shape (M, 4)
        
        deltas = np.zeros(N, dtype=np.int32)
        
        # For each role r in 0,1,2,3:
        # If the other 3 positions are all equal, flipping position r changes the count
        # If the other 3 are all 0: if c[r]==0, mono0 was 1, after flip it's 0 => delta -1
        #                           if c[r]==1, mono0 was 0, after flip it's 1 => delta +1
        # If the other 3 are all 1: if c[r]==1, mono1 was 1, after flip it's 0 => delta -1
        #                           if c[r]==0, mono1 was 0, after flip it's 1 => delta +1
        # Otherwise: no change => delta 0
        
        # Role 0
        others0 = c_aps[:, 1:4]
        eq0 = (others0[:, 0] == others0[:, 1]) & (others0[:, 1] == others0[:, 2])
        # If eq0 and c_aps[:,0] == others0[:,0], flip removes a mono => -1
        # If eq0 and c_aps[:,0] != others0[:,0], flip adds a mono => +1
        contrib0 = np.where(eq0, np.where(c_aps[:, 0] == others0[:, 0], -1, 1), 0)
        np.add.at(deltas, aps_array[:, 0], contrib0)
        
        # Role 1
        others1 = np.column_stack((c_aps[:, 0], c_aps[:, 2], c_aps[:, 3]))
        eq1 = (others1[:, 0] == others1[:, 1]) & (others1[:, 1] == others1[:, 2])
        contrib1 = np.where(eq1, np.where(c_aps[:, 1] == others1[:, 0], -1, 1), 0)
        np.add.at(deltas, aps_array[:, 1], contrib1)
        
        # Role 2
        others2 = np.column_stack((c_aps[:, 0], c_aps[:, 1], c_aps[:, 3]))
        eq2 = (others2[:, 0] == others2[:, 1]) & (others2[:, 1] == others2[:, 2])
        contrib2 = np.where(eq2, np.where(c_aps[:, 2] == others2[:, 0], -1, 1), 0)
        np.add.at(deltas, aps_array[:, 2], contrib2)
        
        # Role 3
        others3 = c_aps[:, 0:3]
        eq3 = (others3[:, 0] == others3[:, 1]) & (others3[:, 1] == others3[:, 2])
        contrib3 = np.where(eq3, np.where(c_aps[:, 3] == others3[:, 0], -1, 1), 0)
        np.add.at(deltas, aps_array[:, 3], contrib3)
        
        return deltas
    
    def compute_pair_deltas(colors, p, q):
        """Compute the change in mono count if we flip both bits p and q simultaneously."""
        # Find APs that contain p or q
        ap_indices = set()
        for ap_idx, role in pos_ap_roles[p]:
            ap_indices.add(ap_idx)
        for ap_idx, role in pos_ap_roles[q]:
            ap_indices.add(ap_idx)
        
        if not ap_indices:
            return 0
        
        ap_list = list(ap_indices)
        ap_arr = np.array(ap_list, dtype=np.int32)
        
        c_before = colors[aps_array[ap_arr]]
        delta_count = 0
        
        for idx in ap_list:
            i0, i1, i2, i3 = aps_array[idx]
            # Determine which positions are p and/or q
            pos_in_ap = [i0, i1, i2, i3]
            flip_mask = [1 if pos in (p, q) else 0 for pos in pos_in_ap]
            
            # Current colors
            cur = [colors[i0], colors[i1], colors[i2], colors[i3]]
            
            # New colors after flipping p and q
            new = [cur[k] ^ flip_mask[k] for k in range(4)]
            
            # Count mono before
            mono_before = 1 if (cur[0] == cur[1] == cur[2] == cur[3]) else 0
            # Count mono after
            mono_after = 1 if (new[0] == new[1] == new[2] == new[3]) else 0
            
            delta_count += mono_after - mono_before
        
        return delta_count
    
    def greedy_polish_vectorized(colors, max_iters=2000):
        """
        Local search using vectorized delta computation with batch flips.
        """
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(colors)
            min_delta = np.min(deltas)
            
            if min_delta < 0:
                # Flip all bits with negative delta
                flip_mask = deltas < 0
                if np.any(flip_mask):
                    colors[flip_mask] = 1 - colors[flip_mask]
                    current = count_mono(colors)
                else:
                    break
            else:
                break
        
        return colors, current
    
    def simulated_annealing_vectorized(initial_colors, start_temp=1.0, end_temp=0.005, 
                            cooling_rate=0.9995, max_steps=30000):
        """
        Simulated annealing using vectorized delta computation for single bit flips.
        """
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(colors)
            
            pos = random.randint(0, N - 1)
            delta = deltas[pos]
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                colors[pos] = 1 - colors[pos]
                current += delta
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count
    
    def pair_flip_search(colors, max_iters=5000):
        """
        Search using pair flips to escape local minima.
        """
        colors = colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try a random pair flip
            p = random.randint(0, N - 1)
            q = random.randint(0, N - 1)
            if p == q:
                continue
            
            delta = compute_pair_deltas(colors, p, q)
            
            if delta < 0:
                colors[p] = 1 - colors[p]
                colors[q] = 1 - colors[q]
                current += delta
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            elif delta == 0:
                # Accept neutral moves with some probability to explore
                if random.random() < 0.1:
                    colors[p] = 1 - colors[p]
                    colors[q] = 1 - colors[q]
        
        return best_colors, best_count
    
    def batched_gradient_descent(colors, max_iters=1000):
        """
        Batched gradient descent: flip all bits that have negative marginal gain.
        This is essentially the same as greedy_polish_vectorized but with a different framing.
        """
        return greedy_polish_vectorized(colors, max_iters)
    
    best_colors = None
    best_score = float('inf')
    
    # Start with a balanced random initialization
    random.seed(42)
    
    # Initial: balanced coloring
    colors_init = np.zeros(N, dtype=np.int8)
    for k in range(N):
        c0_count = np.sum(colors_init[:k] == 0)
        c1_count = np.sum(colors_init[:k] == 1)
        if c0_count <= c1_count:
            colors_init[k] = 0
        else:
            colors_init[k] = 1
    
    polished, score = greedy_polish_vectorized(colors_init)
    if score < best_score:
        best_score = score
        best_colors = polished.copy()
    
    # Multiple restarts with random initializations
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with some bias toward balance
        random_colors = np.random.randint(0, 2, N, dtype=np.int8)
        # Ensure roughly balanced
        num_ones = np.sum(random_colors)
        if num_ones > N // 2 + 5:
            # Flip some 1s to 0s
            ones_indices = np.where(random_colors == 1)[0]
            to_flip = np.random.choice(ones_indices, size=min(10, len(ones_indices)), replace=False)
            random_colors[to_flip] = 0
        elif num_ones < N // 2 - 5:
            zeros_indices = np.where(random_colors == 0)[0]
            to_flip = np.random.choice(zeros_indices, size=min(10, len(zeros_indices)), replace=False)
            random_colors[to_flip] = 1
        
        polished, score = greedy_polish_vectorized(random_colors, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_colors = polished.copy()
        
        if time.time() < deadline - 15:
            # Simulated annealing
            sa_colors, sa_score = simulated_annealing_vectorized(
                polished, start_temp=0.5, end_temp=0.005, 
                cooling_rate=0.9995, max_steps=20000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
        
        if time.time() < deadline - 10:
            # Pair flip search
            pf_colors, pf_score = pair_flip_search(best_colors, max_iters=3000)
            if pf_score < best_score:
                best_score = pf_score
                best_colors = pf_colors.copy()
    
    # Final polish
    if time.time() < deadline - 2:
        final_colors, final_score = greedy_polish_vectorized(best_colors, max_iters=10000)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    return [int(c) for c in best_colors]
