import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Fallback to seed solution if something goes wrong or parameters don't match
    def seed_solution():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if n != 8 or q != 3 or radius != 1:
        return seed_solution()

    try:
        # 1. Generate Ternary Golay Code [12, 6, 6]
        # Generator Matrix (6x12)
        G = [
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1]
        ]
        
        golay_codewords = []
        for coeffs in itertools.product(range(3), repeat=6):
            cw = [0] * 12
            for i in range(6):
                if coeffs[i] != 0:
                    for j in range(12):
                        cw[j] = (cw[j] + coeffs[i] * G[i][j]) % 3
            golay_codewords.append(cw)
            
        # 2. Shorten on first 4 coordinates
        # Group codewords by their prefix (first 4 symbols)
        groups = {}
        for cw in golay_codewords:
            prefix = tuple(cw[:4])
            suffix = cw[4:]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(suffix)
            
        # We have 81 groups, each with 27 codewords of length 8
        
        # Precompute all points in F_3^8 and map to indices
        all_points = list(itertools.product(range(3), repeat=8))
        point_to_index = {p: i for i, p in enumerate(all_points)}
        total_points = len(all_points)
        
        # Precompute the set of indices covered by each group
        group_cover_sets = []
        group_list = []  # Keep list of (prefix, codewords) for retrieval
        for prefix, codewords in groups.items():
            group_list.append((prefix, codewords))
            covered_indices = set()
            for cw in codewords:
                # Add cw itself
                idx = point_to_index[tuple(cw)]
                covered_indices.add(idx)
                # Add neighbors (Hamming distance 1)
                for i in range(8):
                    orig_val = cw[i]
                    for new_val in range(3):
                        if new_val != orig_val:
                            neighbor = list(cw)
                            neighbor[i] = new_val
                            nidx = point_to_index[tuple(neighbor)]
                            covered_indices.add(nidx)
            group_cover_sets.append(covered_indices)
            
        # Greedy Set Cover
        covered_global = set()
        selected_indices = []
        remaining_indices = list(range(len(group_cover_sets)))
        
        while remaining_indices:
            if time.time() > deadline:
                break # Safety margin
                
            best_idx = -1
            best_gain = 0
            for idx in remaining_indices:
                # Calculate gain: how many points in group_cover_sets[idx] are NOT in covered_global
                # Using set difference
                gain = len(group_cover_sets[idx] - covered_global)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1 or best_gain == 0:
                break # No more gain or no groups left
                
            # Select best_idx
            selected_indices.append(best_idx)
            covered_global.update(group_cover_sets[best_idx])
            remaining_indices.remove(best_idx)
            
            if len(covered_global) == total_points:
                break
                
        # Construct final list of codewords
        result = []
        for idx in selected_indices:
            prefix, codewords = group_list[idx]
            for cw in codewords:
                result.append(list(cw))
                
        # Verify validity quickly (optional but safe)
        if result and len(covered_global) == total_points:
            return result
            
        # If greedy failed or didn't cover everything, fallback
        return seed_solution()

    except Exception:
        return seed_solution()
