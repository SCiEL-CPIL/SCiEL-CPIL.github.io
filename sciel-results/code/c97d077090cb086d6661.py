import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(S[j] - S[i])
        return diffs
        
    def is_sidon(S):
        if len(S) < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covered_mask(S):
        diffs = get_diffs(S)
        mask = 0
        for d in diffs:
            if 1 <= d <= 30:
                mask |= (1 << (d - 1))
        return mask

    def beam_search(beam_width=500, max_depth=12, time_budget=30):
        """Beam search to build a Sidon set covering 1..30 with minimal max element."""
        end_time = start_time + time_budget
        
        # State: (sorted_tuple, covered_mask, used_diffs_frozenset)
        # We track used_diffs as a frozenset for fast lookup
        initial_state = ([0], 0, frozenset())
        beam = [initial_state]
        
        best_result = None
        best_max = float('inf')
        
        for depth in range(1, max_depth + 1):
            if time.time() > end_time:
                break
            
            new_beam = []
            seen_states = set()
            
            for S, mask, used_diffs in beam:
                if mask == full_mask:
                    # Found a complete solution
                    max_elem = max(S)
                    if max_elem < best_max:
                        best_max = max_elem
                        best_result = list(S)
                    continue
                
                if len(S) >= 32:
                    continue
                
                # Generate candidates
                candidates = set()
                missing = target_diffs - set(d for d in used_diffs if 1 <= d <= 30)
                
                for s in S:
                    for d in missing:
                        candidates.add(s + d)
                        if s - d > 0:
                            candidates.add(s - d)
                
                # Add some random candidates
                rng = random.Random(hash((S, mask)) % 10000)
                for _ in range(10):
                    s = rng.choice(S)
                    d = rng.randint(1, 60)
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)
                
                # Filter and evaluate candidates
                valid_candidates = []
                for x in candidates:
                    if x <= 0 or x > 10**6:
                        continue
                    if x in S:
                        continue
                    
                    # Check Sidon property
                    new_diffs = set()
                    valid = True
                    for s in S:
                        d = abs(x - s)
                        if d == 0:
                            continue
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    new_used = used_diffs | frozenset(new_diffs)
                    new_mask = mask
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_mask |= (1 << (d - 1))
                    
                    new_S = sorted(S + [x])
                    state_key = (tuple(new_S), new_mask)
                    
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    
                    valid_candidates.append((new_S, new_mask, new_used))
                
                # Sort by: (number of missing diffs, max element)
                valid_candidates.sort(key=lambda item: (
                    len(target_diffs - set(d for d in item[2] if 1 <= d <= 30)),
                    max(item[0])
                ))
                
                # Add top candidates to new beam
                for item in valid_candidates[:beam_width // max(1, len(beam))]:
                    new_beam.append(item)
                    if len(new_beam) >= beam_width:
                        break
            
            # Deduplicate and keep best
            if not new_beam:
                break
            
            # Sort the new beam and keep top beam_width
            new_beam.sort(key=lambda item: (
                len(target_diffs - set(d for d in item[2] if 1 <= d <= 30)),
                max(item[0])
            ))
            beam = new_beam[:beam_width]
        
        return best_result, best_max

    def local_search(S, max_iterations=2000, time_budget=30):
        """Local search to minimize max element while maintaining Sidon property and coverage."""
        end_time = start_time + time_budget
        S = sorted(S)
        
        if not get_diffs(S).issuperset(target_diffs):
            return S
        
        best_S = S[:]
        best_max = max(S)
        
        for _ in range(max_iterations):
            if time.time() > end_time:
                break
            
            improved = False
            n = len(S)
            
            # Try removing each element
            for i in range(n):
                if S[i] == 0:
                    continue
                
                candidate = S[:i] + S[i+1:]
                if len(candidate) < 2:
                    continue
                
                if is_sidon(candidate) and get_diffs(candidate).issuperset(target_diffs):
                    if max(candidate) < best_max:
                        best_S = candidate[:]
                        best_max = max(candidate)
                        S = candidate
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try replacing an element with a smaller one
            for i in range(n):
                if S[i] == 0:
                    continue
                
                # Try values smaller than S[i]
                for new_val in range(1, S[i]):
                    if new_val in S:
                        continue
                    
                    candidate = S[:i] + [new_val] + S[i+1:]
                    candidate.sort()
                    
                    if not is_sidon(candidate):
                        continue
                    
                    if get_diffs(candidate).issuperset(target_diffs):
                        if max(candidate) < best_max:
                            best_S = candidate[:]
                            best_max = max(candidate)
                            S = candidate
                            improved = True
                            break
                
                if improved:
                    break
            
            # Try swapping two elements
            if not improved:
                for i in range(n):
                    if S[i] == 0:
                        continue
                    for j in range(i+1, n):
                        if S[j] == 0:
                            continue
                        
                        # Try swapping S[i] and S[j] with smaller values
                        for new_i in range(1, S[i]):
                            if new_i in S:
                                continue
                            for new_j in range(1, S[j]):
                                if new_j in S:
                                    continue
                                if new_i == new_j:
                                    continue
                                
                                candidate = S[:]
                                candidate[i] = new_i
                                candidate[j] = new_j
                                candidate.sort()
                                
                                if not is_sidon(candidate):
                                    continue
                                
                                if get_diffs(candidate).issuperset(target_diffs):
                                    if max(candidate) < best_max:
                                        best_S = candidate[:]
                                        best_max = max(candidate)
                                        S = candidate
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_S

    # Phase 1: Beam search
    best_set, best_max = beam_search(beam_width=300, max_depth=12, time_budget=40)
    
    if best_set is not None:
        best_set = local_search(best_set, max_iterations=1000, time_budget=30)
        best_max = max(best_set)
    
    # Phase 2: Multiple restarts with different beam widths
    for bw in [100, 200, 400]:
        if time.time() - start_time > deadline - 20:
            break
        
        result, rmax = beam_search(beam_width=bw, max_depth=10, time_budget=20)
        if result is not None and rmax < best_max:
            best_set = result
            best_max = rmax
            best_set = local_search(best_set, max_iterations=500, time_budget=10)
            best_max = max(best_set)
    
    # Phase 3: More aggressive local search if time remains
    if best_set is not None and time.time() - start_time < deadline - 10:
        for _ in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            improved_set = local_search(best_set, max_iterations=300, time_budget=8)
            if max(improved_set) < best_max:
                best_set = improved_set
                best_max = best_max  # already updated
    
    if best_set is not None:
        return best_set
    else:
        # Fallback: Greedy construction
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in sorted(missing):
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S
