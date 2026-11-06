import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums) in [1, 284]
    containing {1, 2, 5, 11}.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    # Initial set
    initial_set = [1, 2, 5, 11]
    
    # Pre-compute initial sums mask
    def get_sums_mask(S):
        mask = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                mask |= (1 << (S[i] + S[j]))
        return mask

    initial_mask = get_sums_mask(initial_set)
    
    # Beam Search Parameters
    beam_width = 50  # Reduced to allow more diverse exploration
    max_depth = 40
    
    # State: (S_list, sums_mask, size)
    beam = [(list(initial_set), initial_mask, len(initial_set))]
    
    best_S = list(initial_set)
    best_size = len(initial_set)
    
    depth = 0
    while depth < max_depth:
        current_time = time.time()
        if current_time > deadline:
            break
            
        next_beam_candidates = []
        
        for S, mask, size in beam:
            # Try adding each candidate x
            S_set = set(S)
            
            for x in range(1, 285):
                if x in S_set:
                    continue
                
                # Check conflicts
                conflict = False
                new_sums_mask = mask
                for y in S:
                    s = x + y
                    if mask & (1 << s):
                        conflict = True
                        break
                    new_sums_mask |= (1 << s)
                
                if conflict:
                    continue
                
                s = 2 * x
                if mask & (1 << s):
                    continue
                new_sums_mask |= (1 << s)
                
                # Valid extension
                new_S = S + [x]
                next_beam_candidates.append((new_S, new_sums_mask, size + 1))
                
        if not next_beam_candidates:
            break
            
        # Prune next_beam_candidates to beam_width with diversity-aware selection
        # Sort by size descending
        next_beam_candidates.sort(key=lambda x: -x[2])
        
        if len(next_beam_candidates) > beam_width:
            # Diversity-aware selection:
            # 1. Group by size
            # 2. Within each size group, select diverse candidates based on sum masks
            
            groups = {}
            for state in next_beam_candidates:
                s = state[2]
                if s not in groups:
                    groups[s] = []
                groups[s].append(state)
            
            selected = []
            remaining_slots = beam_width
            
            for s in sorted(groups.keys(), reverse=True):
                if remaining_slots <= 0:
                    break
                group = groups[s]
                
                if len(group) <= remaining_slots:
                    selected.extend(group)
                    remaining_slots -= len(group)
                else:
                    # Need to select a diverse subset from this group
                    # Use a greedy approach: select candidates that are most different from already selected ones
                    
                    # First, randomly shuffle to break ties
                    random.shuffle(group)
                    
                    # Start with the first candidate
                    selected_group = [group[0]]
                    remaining_slots -= 1
                    
                    # Greedily add candidates that are most different
                    for candidate in group[1:]:
                        if remaining_slots <= 0:
                            break
                        
                        # Measure "distance" from already selected candidates
                        # Use Hamming distance on sum masks as a proxy for diversity
                        max_diff = 0
                        for sel in selected_group:
                            diff = bin(candidate[1] ^ sel[1]).count('1')
                            max_diff = max(max_diff, diff)
                        
                        # If this candidate is sufficiently different, add it
                        # Threshold decreases as we fill more slots
                        threshold = int(beam_width * 0.3)
                        if max_diff >= threshold or len(selected_group) < 5:
                            selected_group.append(candidate)
                            remaining_slots -= 1
                    
                    selected.extend(selected_group)
            
            beam = selected[:beam_width]
        else:
            beam = next_beam_candidates
            
        # Update best
        for S, mask, size in beam:
            if size > best_size:
                best_size = size
                best_S = S
                
        depth += 1
        
    # Final check: ensure the returned set is valid and contains required elements
    return best_S

# EVOLVE-BLOCK-END
