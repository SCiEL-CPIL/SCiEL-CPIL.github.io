import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 75  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Enumerate all 4-sets and precompute the three 4-cycles for each
    all_4sets = list(itertools.combinations(range(N), 4))
    
    def get_cycles(a, b, c, d):
        # Three distinct 4-cycles on vertices a,b,c,d
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def full_validation(colors):
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    # Initialize with a structured coloring: color(i,j) = i
    # This uses 15 colors (0..14)
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
    
    if not full_validation(colors):
        return [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Strategy: Use simulated annealing to maximize distinct colors
    # while maintaining validity. We'll use a soft objective that penalizes
    # rainbow cycles but rewards color diversity.
    
    def count_violations(colors):
        count = 0
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    count += 1
        return count
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Simulated annealing
    current_colors = colors[:]
    current_violations = count_violations(current_colors)
    current_distinct = count_distinct(current_colors)
    
    best_sa_colors = current_colors[:]
    best_sa_distinct = current_distinct
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    num_iterations = 500000
    
    # Precompute for speed: for each edge, which cycles it belongs to
    # and for quick violation checking
    
    for it in range(num_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Temperature
        progress = it / num_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e_idx]
        
        # Choose a new color: either a random color from 0..119,
        # or a color from a neighboring edge (to promote structure)
        if random.random() < 0.3:
            # Pick a color used by some other edge
            other_e = random.randint(0, NUM_EDGES - 1)
            new_color = current_colors[other_e]
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        # Compute delta in violations
        # Only need to check cycles that include e_idx
        delta_viol = 0
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            old_valid = is_valid_cycle(current_colors, cycle)
            current_colors[e_idx] = new_color
            new_valid = is_valid_cycle(current_colors, cycle)
            current_colors[e_idx] = old_color
            if old_valid and not new_valid:
                delta_viol += 1
            elif not old_valid and new_valid:
                delta_viol -= 1
        
        # Compute delta in distinct colors
        old_distinct = current_distinct
        current_colors[e_idx] = new_color
        new_distinct = count_distinct(current_colors)
        current_colors[e_idx] = old_color
        delta_distinct = new_distinct - old_distinct
        
        # Objective: minimize violations, maximize distinct colors
        # delta_obj = delta_viol - delta_distinct (lower is better)
        delta_obj = delta_viol * 100 - delta_distinct
        
        # Accept if improvement, or with probability exp(-delta_obj/T)
        if delta_obj < 0:
            accept = True
        else:
            accept = random.random() < (2.718281828 ** (-delta_obj / T))
        
        if accept:
            current_colors[e_idx] = new_color
            current_violations += delta_viol
            current_distinct = new_distinct
            
            if current_violations == 0 and current_distinct > best_sa_distinct:
                best_sa_distinct = current_distinct
                best_sa_colors = current_colors[:]
    
    # After SA, do a local search to fix any remaining violations
    if count_violations(best_sa_colors) == 0:
        if best_sa_distinct > best_num_colors:
            best_num_colors = best_sa_distinct
            best_colors = best_sa_colors[:]
    else:
        # Try to repair best_sa_colors
        repaired = best_sa_colors[:]
        for _ in range(10000):
            if time.time() - start_time > DEADLINE:
                break
            viol = count_violations(repaired)
            if viol == 0:
                break
            # Find a violating cycle and change one of its edges
            found_fix = False
            for s_idx in range(len(all_4sets)):
                for c_idx in range(3):
                    cycle = cycles_by_4set[s_idx][c_idx]
                    if not is_valid_cycle(repaired, cycle):
                        # Change one edge in this cycle
                        e_to_change = random.choice(cycle)
                        old_c = repaired[e_to_change]
                        # Try to match one of the other edges' colors
                        other_colors = [repaired[e] for e in cycle if e != e_to_change]
                        new_c = random.choice(other_colors)
                        repaired[e_to_change] = new_c
                        found_fix = True
                        break
                if found_fix:
                    break
            if not found_fix:
                break
        
        if count_violations(repaired) == 0:
            d = count_distinct(repaired)
            if d > best_num_colors:
                best_num_colors = d
                best_colors = repaired[:]
    
    # Final greedy improvement: try to add more colors
    # Sort edges by constraint count
    edge_constraint_counts = [(len(edge_to_4sets[e]), e) for e in range(NUM_EDGES)]
    edge_constraint_counts.sort()
    sorted_edges = [e for _, e in edge_constraint_counts]
    
    used_colors = set(best_colors)
    
    for new_color in range(15, 120):
        if time.time() - start_time > DEADLINE:
            break
        if new_color in used_colors:
            continue
        
        found = False
        for e_idx in sorted_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            old_color = best_colors[e_idx]
            best_colors[e_idx] = new_color
            
            # Check validity
            valid = True
            for (s_idx, c_idx) in edge_to_4sets[e_idx]:
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(best_colors, cycle):
                    valid = False
                    break
            
            if valid:
                found = True
                used_colors.add(new_color)
                break
            
            best_colors[e_idx] = old_color
        
        if not found:
            for e_idx in reversed(sorted_edges):
                if time.time() - start_time > DEADLINE:
                    break
                
                old_color = best_colors[e_idx]
                best_colors[e_idx] = new_color
                
                valid = True
                for (s_idx, c_idx) in edge_to_4sets[e_idx]:
                    cycle = cycles_by_4set[s_idx][c_idx]
                    if not is_valid_cycle(best_colors, cycle):
                        valid = False
                        break
                
                if valid:
                    found = True
                    used_colors.add(new_color)
                    break
                
                best_colors[e_idx] = old_color
    
    # Final validation
    if full_validation(best_colors):
        return best_colors
    else:
        # Fallback to the initial valid coloring
        return colors

# Run construct once at module level to ensure it works
# Actually, we should not call it at module level, just define it
