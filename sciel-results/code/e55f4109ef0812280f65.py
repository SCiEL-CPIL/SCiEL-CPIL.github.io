import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_bitset_structures(target_n):
        """
        Build bitset structures for fast violation counting.
        For each triple, store:
        - quad_mask: for each other triple in a quad, which quads it belongs to
        - Use bitsets to count violations quickly
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # For each triple, store the list of quads it's in, and for each quad, the other 3 triples
        # quad_info[i] = list of (other_triple_indices, quad_id)
        quad_info = [[] for _ in range(num_triples)]
        
        quad_id = 0
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            
            quad_info[t1].append((t2, t3, t4, quad_id))
            quad_info[t2].append((t1, t3, t4, quad_id))
            quad_info[t3].append((t1, t2, t4, quad_id))
            quad_info[t4].append((t1, t2, t3, quad_id))
            quad_id += 1
        
        # For each triple, store the list of quints it's in, and for each quint, the other 9 triples
        quint_info = [[] for _ in range(num_triples)]
        
        quint_id = 0
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            
            for i, t_idx in enumerate(triple_indices_in_quint):
                others = [triple_indices_in_quint[j] for j in range(10) if j != i]
                quint_info[t_idx].append((others, quint_id))
                quint_id += 1
        
        return triple_list, num_triples, triple_to_idx, quad_info, quint_info

    def count_violations_bitset(n, colors, triple_list, triple_to_idx, quad_info, quint_info):
        """
        Count red K4 and blue K5 violations using bitsets.
        colors is a numpy array of 0/1.
        """
        num_triples = len(triple_list)
        
        # For red K4: a quad is all-red if all 4 triples are red (color 0)
        # For blue K5: a quint is all-blue if all 10 triples are blue (color 1)
        
        red_k4_count = 0
        blue_k5_count = 0
        
        # Process quads
        seen_quads = set()
        for t_idx in range(num_triples):
            for (o1, o2, o3, qid) in quad_info[t_idx]:
                if qid in seen_quads:
                    continue
                seen_quads.add(qid)
                if colors[t_idx] == 0 and colors[o1] == 0 and colors[o2] == 0 and colors[o3] == 0:
                    red_k4_count += 1
        
        # Process quints
        seen_quints = set()
        for t_idx in range(num_triples):
            for (others, qid) in quint_info[t_idx]:
                if qid in seen_quints:
                    continue
                seen_quints.add(qid)
                if colors[t_idx] == 1:
                    all_blue = True
                    for o in others:
                        if colors[o] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        blue_k5_count += 1
        
        return red_k4_count, blue_k5_count

    def run_local_search_bitset(target_n, initial_colors, deadline, seed_offset=0):
        """
        Runs simulated annealing with bitset-accelerated violation counting.
        Returns (colors, num_violations)
        """
        triple_list, num_triples, triple_to_idx, quad_info, quint_info = build_bitset_structures(target_n)
        
        # Color array: 0 for Red, 1 for Blue
        colors = np.array(initial_colors, dtype=np.int8)
        
        # Initial violation count
        red_k4_count, blue_k5_count = count_violations_bitset(target_n, colors, triple_list, triple_to_idx, quad_info, quint_info)
        current_cost = red_k4_count + blue_k5_count
        best_cost = current_cost
        best_colors = colors.copy()
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.001
        cooling = 0.995
        
        iter_count = 0
        while time.time() < deadline - 5:
            # Choose a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            
            # Calculate delta red_k4
            delta_r4 = 0
            if old_color == 0:  # Currently Red, flipping to Blue
                for (o1, o2, o3, qid) in quad_info[t_idx]:
                    if colors[o1] == 0 and colors[o2] == 0 and colors[o3] == 0:
                        delta_r4 -= 1
            else:  # Currently Blue, flipping to Red
                for (o1, o2, o3, qid) in quad_info[t_idx]:
                    if colors[o1] == 0 and colors[o2] == 0 and colors[o3] == 0:
                        delta_r4 += 1
                        
            # Calculate delta blue_k5
            delta_b5 = 0
            if old_color == 1:  # Currently Blue, flipping to Red
                for (others, qid) in quint_info[t_idx]:
                    all_blue = True
                    for o in others:
                        if colors[o] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        delta_b5 -= 1
            else:  # Currently Red, flipping to Blue
                for (others, qid) in quint_info[t_idx]:
                    all_blue = True
                    for o in others:
                        if colors[o] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        delta_b5 += 1
                        
            delta_cost = delta_r4 + delta_b5
            
            # Accept?
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Apply flip
                colors[t_idx] = 1 - old_color
                red_k4_count += delta_r4
                blue_k5_count += delta_b5
                current_cost = red_k4_count + blue_k5_count
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist(), 0
            
            T *= cooling
            if T < T_min:
                T = T_min
                if iter_count % 10000 == 0:
                    T = 5.0  # Reheat
            
            iter_count += 1
            
        return best_colors.tolist(), best_cost

    def backtrack_with_forward_checking(n, deadline, seed_offset=0):
        """
        Backtracking search with forward checking for guaranteed valid colorings.
        Works well for smaller n.
        """
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Build quad and quint info
        quad_info = [[] for _ in range(num_triples)]
        for a, b, c, d in combinations(range(n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            quad_info[t1].append((t2, t3, t4))
            quad_info[t2].append((t1, t3, t4))
            quad_info[t3].append((t1, t2, t4))
            quad_info[t4].append((t1, t2, t3))
        
        quint_info = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            for i, t_idx in enumerate(triple_indices_in_quint):
                others = [triple_indices_in_quint[j] for j in range(10) if j != i]
                quint_info[t_idx].append(others)
        
        colors = [-1] * num_triples  # -1 = uncolored
        
        # Count how many red triples each quad has, and blue triples each quint has
        quad_red_count = [0] * 1  # Will be rebuilt
        quint_blue_count = [0] * 1
        
        # Precompute all quads and quints
        all_quads = []
        for a, b, c, d in combinations(range(n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            all_quads.append([t1, t2, t3, t4])
        
        all_quints = []
        for quint in combinations(range(n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            all_quints.append(triple_indices_in_quint)
        
        num_quads = len(all_quads)
        num_quints = len(all_quints)
        
        quad_red = [0] * num_quads
        quint_blue = [0] * num_quints
        
        def is_consistent():
            """Check if current partial coloring is consistent (no completed violations)."""
            for q in all_quads:
                if quad_red[q[0]] + quad_red[q[1]] + quad_red[q[2]] + quad_red[q[3]] >= 4:
                    # All 4 are red
                    return False
            for q in all_quints:
                if quint_blue[q[0]] + quint_blue[q[1]] + quint_blue[q[2]] + quint_blue[q[3]] + \
                   quint_blue[q[4]] + quint_blue[q[5]] + quint_blue[q[6]] + quint_blue[q[7]] + \
                   quint_blue[q[8]] + quint_blue[q[9]] >= 10:
                    # All 10 are blue
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_triples:
                return True
            
            # Try red (0) first
            colors[idx] = 0
            # Update quad_red counts
            quad_updates = []
            for (o1, o2, o3) in quad_info[idx]:
                # Find which quads this triple is in
                pass  # This is getting complex, let's use a simpler approach
            
            # Simpler: just check consistency after assignment
            # But we need to track counts properly
            
            # Let's use a different approach: maintain counts
            # For now, just do simple backtracking with consistency check
            if is_consistent():
                if backtrack(idx + 1):
                    return True
            
            # Try blue (1)
            colors[idx] = 1
            if is_consistent():
                if backtrack(idx + 1):
                    return True
            
            colors[idx] = -1
            return False
        
        # This approach is too slow for larger n. Let's use a smarter heuristic.
        # Order triples by degree (number of constraints)
        triple_degree = [len(quad_info[i]) + len(quint_info[i]) for i in range(num_triples)]
        order = sorted(range(num_triples), key=lambda x: -triple_degree[x])
        
        # Use random restarts with greedy initialization
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Greedy initialization with random tie-breaking
            colors = [-1] * num_triples
            quad_red = [0] * num_quads
            quint_blue = [0] * num_quints
            
            for idx in order:
                if time.time() > deadline:
                    break
                
                # Try both colors, pick the one that causes fewer violations
                best_color = -1
                best_violations = float('inf')
                
                for c in [0, 1]:
                    colors[idx] = c
                    # Count violations
                    violations = 0
                    for (o1, o2, o3) in quad_info[idx]:
                        if c == 0 and colors[o1] == 0 and colors[o2] == 0 and colors[o3] == 0:
                            violations += 1
                    for others in quint_info[idx]:
                        if c == 1 and all(colors[o] == 1 for o in others):
                            violations += 1
                    
                    if violations < best_violations:
                        best_violations = violations
                        best_color = c
                
                colors[idx] = best_color
            
            if all(c != -1 for c in colors):
                # Check if valid
                if is_valid(n, colors):
                    return colors
        
        return None

    # Strategy: Try increasing n
    # Focus on larger n first since that's what we want to maximize
    
    # For n <= 20, use backtracking (should be fast)
    # For n > 20, use local search
    
    best_found = (9, mod_color(9, 3))
    
    # Try backtracking for small n first
    for n in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 30:
            break
        
        alloc = 5 if n <= 15 else 3
        current_deadline = time.time() + alloc
        
        result = backtrack_with_forward_checking(n, current_deadline)
        if result is not None:
            best_found = (n, result)
    
    # Now try local search for larger n
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline - 15:
            break
        if n <= best_found[0]:
            continue
        
        # Allocate time based on n
        if n >= 30:
            alloc = 15
        elif n >= 25:
            alloc = 10
        else:
            alloc = 8
        
        current_deadline = time.time() + alloc
        
        # Multiple restarts with different initializations
        max_restarts = 20
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline:
                break
            
            # Generate initial coloring
            if r == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 1:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 2:
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 4:
                init = [0 if (a*2 + b*3 + c*5) % 11 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 5:
                init = [0 if (a*3 + b*5 + c*7) % 13 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            else:
                # Random with bias
                p = 0.3 + 0.4 * (r % 5) / 5.0
                init = [0 if random.random() < p else 1 for _ in combinations(range(n), 3)]
            
            colors, cost = run_local_search_bitset(n, init, current_deadline, seed_offset=r)
            
            if cost == 0:
                # Verify strictly
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break
    
    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
