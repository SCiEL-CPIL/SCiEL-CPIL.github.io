def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    mult_solution = try_multiplicative_scheme()
    if mult_solution is not None:
        return mult_solution

    init_colors = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Frequency array for colors 0..19
    max_color_val = 20
    freq = [0] * max_color_val
    for c in init_colors:
        freq[c] += 1
        
    def get_num_colors(freq):
        return sum(1 for f in freq if f > 0)
        
    best_colors = init_colors.copy()
    best_score = calculate_score(best_colors)
    best_num_colors = get_num_colors(freq)
    
    current_colors = best_colors.copy()
    current_freq = freq.copy()
    current_score = best_score
    current_num_colors = best_num_colors
    
    # Precompute penalty contribution per edge? 
    # Too complex. Just recalc validity penalty.
    # But we can optimize validity penalty calculation?
    # The validity penalty is the bottleneck.
    # Let's stick to the list comprehension in calculate_score, but maybe cache it?
    # No, colors change every step.
    
    # To speed up validity check, we can check only affected K4s?
    # An edge change affects K4s containing that edge.
    # Number of K4s containing a specific edge in K16:
    # Edge (u,v). Need 2 more vertices from remaining 14.
    # C(14, 2) = 91 K4s.
    # So we only need to re-evaluate 91 K4s instead of 1820.
    # This is a 20x speedup!
    
    # Precompute edge_to_k4s mapping
    edge_to_k4s = [[] for _ in range(NUM_EDGES)]
    for k4_idx, e_idxs in enumerate(K4_EDGE_INDICES):
        for e_idx in e_idxs:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Current validity penalty
    # We need to maintain the penalty for each K4?
    # Or just sum?
    # If we maintain sum, we need to know the penalty of the 91 affected K4s before and after.
    
    current_validity_pen = 0
    # Calculate initial validity penalty (should be 0)
    for e_idxs in K4_EDGE_INDICES:
        distinct = len({current_colors[e_idxs[0]], current_colors[e_idxs[1]], current_colors[e_idxs[2]], current_colors[e_idxs[3]], current_colors[e_idxs[4]], current_colors[e_idxs[5]]})
        if distinct < 5:
            current_validity_pen += (5 - distinct)
            
    current_score = current_validity_pen * 1000 + current_num_colors
    best_score = current_score
    
    T = 50.0
    T_min = 1.0
    alpha = 0.999
    t_start = time.time()
    
    while time.time() < deadline:
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e_idx]
        new_color = random.randint(0, 16)
        
        if new_color == old_color:
            continue
            
        # Calculate change in validity penalty
        # Affected K4s are those containing e_idx
        affected_k4s = edge_to_k4s[e_idx]
        
        delta_pen = 0
        # Store old distinct counts for affected K4s?
        # We can just compute old and new penalty for these K4s.
        
        old_pen_sum = 0
        for k4_idx in affected_k4s:
            e_idxs = K4_EDGE_INDICES[k4_idx]
            # Get colors
            c = [current_colors[ei] for ei in e_idxs]
            distinct = len(set(c))
            if distinct < 5:
                old_pen_sum += (5 - distinct)
                
        # Apply change
        current_colors[e_idx] = new_color
        current_freq[old_color] -= 1
        current_freq[new_color] += 1
        
        new_pen_sum = 0
        for k4_idx in affected_k4s:
            e_idxs = K4_EDGE_INDICES[k4_idx]
            c = [current_colors[ei] for ei in e_idxs]
            distinct = len(set(c))
            if distinct < 5:
                new_pen_sum += (5 - distinct)
                
        delta_pen = new_pen_sum - old_pen_sum
        
        # Calculate change in num_colors
        old_num_colors = current_num_colors
        if current_freq[old_color] == 0:
            old_num_colors -= 1
        if current_freq[new_color] == 1:
            old_num_colors += 1
        new_num_colors = old_num_colors
        
        new_score = (current_validity_pen + delta_pen) * 1000 + new_num_colors
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            # Accept
            current_validity_pen += delta_pen
            current_num_colors = new_num_colors
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_colors = current_colors.copy()
        else:
            # Reject
            current_colors[e_idx] = old_color
            current_freq[old_color] += 1
            current_freq[new_color] -= 1
            
        T *= alpha
        if T < T_min:
            T = T_min

    if current_validity_pen > 0:
         return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    return best_colors
