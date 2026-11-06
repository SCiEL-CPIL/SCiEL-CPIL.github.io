"""Combined program: density-incremental greedy construction with enhanced local search."""
import time
import random
import math

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute APs ending at each index x (1-indexed)
    # APs are (i, j, k, x) such that i, j, k, x form a 4-AP with i = x-3d
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            i = x - 3 * d
            if i >= 1:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))

    # Precompute all 4-APs in 0-indexed form for local search
    aps_all = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps_all.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps_all)
    
    # For each position, precompute which APs it participates in (0-indexed)
    pos_aps = [[] for _ in range(N)]
    for idx, (i, j, k, l) in enumerate(aps_all):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)

    def ap_is_violation_0(colors, idx):
        i, j, k, l = aps_all[idx]
        return len({colors[i], colors[j], colors[k], colors[l]}) < 3

    def compute_violation_count_0(colors):
        count = 0
        for idx in range(num_aps):
            if ap_is_violation_0(colors, idx):
                count += 1
        return count

    def compute_violation_set_0(colors):
        vs = set()
        for idx in range(num_aps):
            if ap_is_violation_0(colors, idx):
                vs.add(idx)
        return vs

    def try_merge_colors_0(colors, deadline):
        """Try to merge color pairs to reduce distinct color count while maintaining validity."""
        current_colors = colors[:]
        current_distinct = len(set(current_colors))
        
        if current_distinct <= 4:
            return current_colors, current_distinct
        
        # Try merging each pair of colors
        color_list = sorted(set(current_colors))
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(color_list)):
                for j in range(i + 1, len(color_list)):
                    if time.time() > deadline - 5:
                        return current_colors, len(set(current_colors))
                    
                    c1 = color_list[i]
                    c2 = color_list[j]
                    
                    # Try merging c2 into c1
                    test_colors = current_colors[:]
                    for idx in range(N):
                        if test_colors[idx] == c2:
                            test_colors[idx] = c1
                    
                    # Check if still valid
                    if compute_violation_count_0(test_colors) == 0:
                        current_colors = test_colors
                        color_list = sorted(set(current_colors))
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_colors, len(set(current_colors))

    def density_greedy_construct(k_max=8, deadline=None):
        """
        Density-incremental greedy construction.
        Iteratively assign colors to positions 1..N, choosing the color that
        minimizes the increase in 'critical' APs (APs at risk of having <3 colors).
        Dynamically expand palette only when no existing color works.
        """
        if deadline is None:
            deadline = start_time + 80.0
        
        colors = [None] * (N + 1)
        colors[1] = 0
        
        # Track which APs are "critical" - i.e., currently have < 3 distinct colors
        # among their assigned positions
        critical_aps = set()
        
        for x in range(2, N + 1):
            if time.time() > deadline - 3:
                return None
            
            # Find all APs that include position x and have other positions already colored
            relevant_aps = []
            for (i, j, k) in aps_ending[x]:
                # Check if at least one of i,j,k is colored
                if colors[i] is not None or colors[j] is not None or colors[k] is not None:
                    relevant_aps.append((i, j, k))
            
            best_color = None
            best_score = float('inf')
            
            # Try existing colors first (0 to k_max-1, but only those actually used)
            used_colors = set(c for c in colors[1:x] if c is not None)
            if not used_colors:
                used_colors = {0}
            
            for c in sorted(used_colors):
                # Compute score: number of new critical APs created
                new_critical = 0
                for (i, j, k) in relevant_aps:
                    # Check if this AP becomes critical with color c at x
                    assigned_colors = []
                    if colors[i] is not None:
                        assigned_colors.append(colors[i])
                    if colors[j] is not None:
                        assigned_colors.append(colors[j])
                    if colors[k] is not None:
                        assigned_colors.append(colors[k])
                    assigned_colors.append(c)
                    
                    distinct = len(set(assigned_colors))
                    # If fewer than 3 distinct colors, this is critical
                    if distinct < 3:
                        new_critical += 1
                
                score = new_critical
                if score < best_score:
                    best_score = score
                    best_color = c
            
            # If no existing color works well, try a new color
            if best_score > 0:
                new_color_id = max(used_colors) + 1 if used_colors else 0
                if new_color_id < k_max:
                    # Test new color
                    new_critical = 0
                    for (i, j, k) in relevant_aps:
                        assigned_colors = []
                        if colors[i] is not None:
                            assigned_colors.append(colors[i])
                        if colors[j] is not None:
                            assigned_colors.append(colors[j])
                        if colors[k] is not None:
                            assigned_colors.append(colors[k])
                        assigned_colors.append(new_color_id)
                        
                        distinct = len(set(assigned_colors))
                        if distinct < 3:
                            new_critical += 1
                    
                    if new_critical < best_score:
                        best_score = new_critical
                        best_color = new_color_id
            
            if best_color is None:
                return None
            
            colors[x] = best_color
            
            # Update critical_aps
            for (i, j, k) in relevant_aps:
                assigned = []
                if colors[i] is not None:
                    assigned.append(colors[i])
                if colors[j] is not None:
                    assigned.append(colors[j])
                if colors[k] is not None:
                    assigned.append(colors[k])
                if colors[x] is not None:
                    assigned.append(colors[x])
                
                if len(assigned) == 4:
                    if len(set(assigned)) < 3:
                        critical_aps.add((i, j, k, x))
                    else:
                        critical_aps.discard((i, j, k, x))
        
        # Verify the solution
        if any(c is None for c in colors[1:]):
            return None
        
        sol = colors[1:]
        if compute_violation_count_0(sol) == 0:
            return sol
        
        return None

    def local_search_0(colors, max_k, deadline, temperature=0.1):
        """Local search with simulated annealing-inspired acceptance."""
        colors = colors[:]
        viol_set = compute_violation_set_0(colors)
        cur_viol = len(viol_set)
        
        if cur_viol == 0:
            return colors, 0
        
        best_local = colors[:]
        best_local_viol = cur_viol
        no_improve = 0
        max_steps = 8000
        t = temperature
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if cur_viol == 0:
                return colors, 0
            
            # Pick position biased toward violation-heavy positions
            if random.random() < 0.8 and viol_set:
                # Pick a random violating AP, then a random position in it
                viol_ap_idx = random.choice(list(viol_set))
                i, j, k, l = aps_all[viol_ap_idx]
                pos = random.choice([i, j, k, l])
            else:
                pos = random.randint(0, N - 1)
            
            old_color = colors[pos]
            
            # Choose new color: prefer colors that appear in violating APs
            if random.random() < 0.4:
                new_color = random.randint(0, max_k - 1)
            else:
                # Sample from colors present in positions of violating APs
                candidate_colors = set()
                for ap_idx in list(viol_set)[:10]:
                    ii, jj, kk, ll = aps_all[ap_idx]
                    candidate_colors.add(colors[ii])
                    candidate_colors.add(colors[jj])
                    candidate_colors.add(colors[kk])
                    candidate_colors.add(colors[ll])
                if candidate_colors:
                    new_color = random.choice(list(candidate_colors))
                else:
                    new_color = random.randint(0, max_k - 1)
            
            if new_color == old_color:
                continue
            
            # Compute delta in violation count
            delta = 0
            affected_aps = pos_aps[pos]
            for ap_idx in affected_aps:
                was_viol = ap_idx in viol_set
                colors[pos] = new_color
                is_viol = ap_is_violation_0(colors, ap_idx)
                colors[pos] = old_color
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            
            # Acceptance criterion: always accept improvements, sometimes accept neutral/worse
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.6
            else:
                # Simulated annealing: accept worse moves with probability exp(-delta/t)
                if t > 0.01:
                    prob = math.exp(-delta / t)
                    accept = random.random() < prob
            
            if accept:
                colors[pos] = new_color
                for ap_idx in affected_aps:
                    was_viol = ap_idx in viol_set
                    is_viol = ap_is_violation_0(colors, ap_idx)
                    if was_viol and not is_viol:
                        viol_set.discard(ap_idx)
                    elif not was_viol and is_viol:
                        viol_set.add(ap_idx)
                cur_viol = len(viol_set)
                
                if cur_viol < best_local_viol:
                    best_local_viol = cur_viol
                    best_local = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Cool down
                t = max(0.001, t * 0.9995)
            else:
                no_improve += 1
            
            if no_improve > 1500:
                break
        
        if best_local_viol < cur_viol:
            return best_local, best_local_viol
        return colors, cur_viol

    best_colors = None
    best_distinct = 128

    # First, try density-incremental greedy construction
    for k_max in range(3, 9):
        if time.time() > deadline - 10:
            break
        
        sol = density_greedy_construct(k_max, deadline)
        if sol:
            used = sorted(set(sol))
            mapping = {c: i for i, c in enumerate(used)}
            candidate = [mapping[c] for c in sol]
            distinct = len(used)
            # Try to merge colors further
            merged, distinct = try_merge_colors_0(candidate, deadline)
            if distinct < best_distinct:
                best_distinct = distinct
                best_colors = merged
            if best_distinct <= 4:
                return best_colors

    # If greedy didn't find a good solution, try local search with various initializations
    if time.time() < deadline - 10:
        # Try different numbers of colors
        for max_k in range(4, 10):
            if time.time() > deadline - 5:
                break
            
            for trial in range(40):
                if time.time() > deadline - 3:
                    break
                
                # Random initial coloring
                colors = [random.randint(0, max_k - 1) for _ in range(N)]
                
                # Run local search
                colors, viol_count = local_search_0(colors, max_k, deadline)
                
                if viol_count == 0:
                    # Try to merge colors
                    merged_colors, distinct = try_merge_colors_0(colors, deadline)
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = merged_colors[:]
                        if best_distinct <= 4:
                            return best_colors

    # Fallback: seed program logic
    if best_colors is not None:
        return best_colors

    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                    for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                            for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
