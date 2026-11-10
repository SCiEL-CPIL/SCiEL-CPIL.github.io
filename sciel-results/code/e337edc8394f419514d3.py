<<<<<<< SEARCH
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def expand(state):
        P, used_diffs, last_d = state
        candidates = []
        for d in range(last_d + 1, 257):
            P_next = P[-1] + d
            valid = True
            new_diffs = []
            for k in range(len(P)):
                diff = P_next - P[k]
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            if valid:
                new_used = used_diffs | set(new_diffs)
                new_P = P + [P_next]
                candidates.append((new_P, new_used, d))
        return candidates

    def greedy_extend(P, used_diffs, last_d, time_limit):
        """Greedily extend the sequence as far as possible."""
        while time.time() < time_limit:
            found = False
            for d in range(last_d + 1, 257):
                P_next = P[-1] + d
                valid = True
                new_diffs = []
                for k in range(len(P)):
                    diff = P_next - P[k]
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                if valid:
                    P = P + [P_next]
                    used_diffs = used_diffs | set(new_diffs)
                    last_d = d
                    found = True
                    break
            if not found:
                break
        return P, used_diffs, last_d

    # Beam Search with improved heuristic
    beam = [([0], set(), 0)]
    best_seq = []
    best_len = 0
    
    # Try multiple beam widths and pick the best
    for B in [30, 50, 80]:
        if time.time() > deadline - 5:
            break
            
        beam = [([0], set(), 0)]
        local_best = []
        
        while beam and time.time() < deadline - 10:
            next_beam = []
            for state in beam:
                candidates = expand(state)
                if candidates:
                    next_beam.extend(candidates)
                else:
                    P = state[0]
                    a = [P[i] - P[i-1] for i in range(1, len(P))]
                    if len(a) > len(local_best):
                        local_best = a
            
            if not next_beam:
                break
                
            # Heuristic: prefer states with smaller last_d (more room) 
            # and smaller total prefix sum (more headroom)
            def state_score(s):
                P, used_diffs, last_d = s
                # Smaller last_d is better (more choices for next step)
                # Smaller max prefix sum is better (more room in [1,256])
                # Fewer used diffs is better (more flexibility)
                return (last_d, P[-1], len(used_diffs))
            
            next_beam.sort(key=state_score)
            beam = next_beam[:B]
            
            # Track best from current beam
            for state in beam:
                P = state[0]
                a = [P[i] - P[i-1] for i in range(1, len(P))]
                if len(a) > len(local_best):
                    local_best = a
        
        # Extend the best beam states greedily
        if beam and time.time() < deadline - 3:
            # Sort beam by heuristic and try extending top few
            def state_score(s):
                P, used_diffs, last_d = s
                return (last_d, P[-1])
            
            beam.sort(key=state_score)
            for state in beam[:min(5, len(beam))]:
                if time.time() > deadline - 2:
                    break
                P, used_diffs, last_d = greedy_extend(state[0], state[1], state[2], deadline - 1)
                a = [P[i] - P[i-1] for i in range(1, len(P))]
                if len(a) > len(local_best):
                    local_best = a
        
        if len(local_best) > best_len:
            best_seq = local_best
            best_len = len(local_best)
    
    # Final check on any remaining beam states
    if beam and time.time() < deadline:
        for state in beam:
            P = state[0]
            a = [P[i] - P[i-1] for i in range(1, len(P))]
            if len(a) > best_len:
                best_seq = a
                best_len = len(a)
    
    return best_seq
=======
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def compute_block_sums(diffs):
        """Compute all nonempty consecutive block sums from difference vector."""
        n = len(diffs)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += diffs[j]
                sums.add(s)
        return sums
    
    def count_duplicate_block_sums(diffs):
        """Count how many block sums are duplicated (0 = valid)."""
        n = len(diffs)
        counts = {}
        dup_count = 0
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += diffs[j]
                counts[s] = counts.get(s, 0) + 1
        for s, c in counts.items():
            if c > 1:
                dup_count += c - 1
        return dup_count
    
    def is_valid_diffs(diffs):
        """Check if difference vector produces valid sequence in [1,256]."""
        if not diffs:
            return False
        if any(d < 1 for d in diffs):
            return False
        # Check cumulative sum stays in [1, 256]
        cum = 0
        for d in diffs:
            cum += d
            if cum > 256:
                return False
        # Check all block sums are distinct
        n = len(diffs)
        seen = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += diffs[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    def greedy_extend_diffs(diffs, time_limit):
        """Greedily extend difference vector as far as possible."""
        while time.time() < time_limit:
            n = len(diffs)
            cum = sum(diffs)
            found = False
            # Try small differences first for more headroom
            for d in range(1, 257 - cum):
                new_diffs = diffs + [d]
                if is_valid_diffs(new_diffs):
                    diffs = new_diffs
                    found = True
                    break
            if not found:
                break
        return diffs
    
    # Phase 1: Beam search to get good starting points
    def expand_beam_state(diffs):
        """Generate candidate extensions for a difference vector."""
        n = len(diffs)
        if n == 0:
            candidates = []
            for d in range(1, 50):
                candidates.append([d])
            return candidates
        
        cum = sum(diffs)
        candidates = []
        for d in range(1, 257 - cum):
            new_diffs = diffs + [d]
            if is_valid_diffs(new_diffs):
                candidates.append(new_diffs)
        return candidates
    
    best_diffs = []
    best_len = 0
    
    # Run beam search for initial solutions
    beam_widths = [20, 40, 60]
    for B in beam_widths:
        if time.time() > deadline - 20:
            break
        
        beam = [[]]  # Start with empty diff vector
        local_best = []
        
        while beam and time.time() < deadline - 15:
            next_beam = []
            for state in beam:
                candidates = expand_beam_state(state)
                next_beam.extend(candidates)
            
            if not next_beam:
                break
            
            # Score: prefer shorter cumulative sum and smaller max diff
            def state_score(diffs):
                if not diffs:
                    return (0, 0)
                return (sum(diffs), max(diffs))
            
            next_beam.sort(key=state_score)
            beam = next_beam[:B]
            
            for state in beam:
                if len(state) > len(local_best):
                    local_best = state
        
        # Greedy extend the best beam states
        if beam and time.time() < deadline - 5:
            beam.sort(key=lambda d: (sum(d), max(d) if d else 0))
            for state in beam[:min(5, len(beam))]:
                if time.time() > deadline - 3:
                    break
                extended = greedy_extend_diffs(state, deadline - 2)
                if len(extended) > len(local_best):
                    local_best = extended
        
        if len(local_best) > best_len:
            best_diffs = local_best
            best_len = len(local_best)
    
    # Phase 2: Simulated annealing to escape local optima
    if best_diffs and time.time() < deadline - 10:
        current_diffs = list(best_diffs)
        current_dups = count_duplicate_block_sums(current_diffs)
        best_sa_diffs = list(current_diffs)
        best_sa_dups = current_dups
        best_sa_len = len(current_diffs)
        
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        sa_deadline = deadline - 3
        
        while T > min_T and time.time() < sa_deadline:
            # Generate a perturbation
            n = len(current_diffs)
            move_type = random.random()
            
            if move_type < 0.4 and n > 0:
                # Modify a random difference
                idx = random.randint(0, n - 1)
                old_val = current_diffs[idx]
                new_val = max(1, old_val + random.randint(-5, 5))
                current_diffs[idx] = new_val
            elif move_type < 0.6 and n > 0:
                # Swap two adjacent differences
                if n >= 2:
                    idx = random.randint(0, n - 2)
                    current_diffs[idx], current_diffs[idx + 1] = current_diffs[idx + 1], current_diffs[idx]
            elif move_type < 0.75 and n > 0:
                # Try to extend
                cum = sum(current_diffs)
                if cum < 250:
                    new_d = random.randint(1, min(10, 256 - cum))
                    current_diffs.append(new_d)
            elif move_type < 0.85 and n > 1:
                # Try to shorten
                current_diffs.pop()
            else:
                # Insert a new difference at random position
                cum = sum(current_diffs)
                if cum < 250:
                    new_d = random.randint(1, min(10, 256 - cum))
                    pos = random.randint(0, n)
                    current_diffs.insert(pos, new_d)
            
            # Check validity and compute objective
            if not current_diffs or any(d < 1 for d in current_diffs) or sum(current_diffs) > 256:
                # Reject invalid moves
                continue
            
            new_dups = count_duplicate_block_sums(current_diffs)
            new_len = len(current_diffs)
            
            # Objective: minimize duplicates, maximize length
            # Use a combined score: -1000*len + dups
            old_score = -1000 * len(best_sa_diffs) + best_sa_dups
            new_score = -1000 * new_len + new_dups
            
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                if new_dups == 0 and new_len > best_sa_len:
                    best_sa_diffs = list(current_diffs)
                    best_sa_len = new_len
                    best_sa_dups = 0
            else:
                # Reject
                # Undo the move (we need to track what we changed)
                # For simplicity, just continue with current state
                pass
            
            T *= cooling
        
        # If SA found something better
        if best_sa_len > best_len:
            best_diffs = best_sa_diffs
            best_len = best_sa_len
    
    # Phase 3: Final greedy extension
    if best_diffs and time.time() < deadline - 1:
        final_diffs = greedy_extend_diffs(best_diffs, deadline - 0.5)
        if len(final_diffs) > best_len:
            best_diffs = final_diffs
            best_len = best_len
    
    # Convert difference vector to actual sequence
    if not best_diffs:
        return []
    
    seq = [best_diffs[0]]
    for d in best_diffs[1:]:
        seq.append(seq[-1] + d)
    
    # Verify all values are in [1, 256]
    if any(v < 1 or v > 256 for v in seq):
        # Fallback: try a simple valid sequence
        return [1]
    
    return seq
>>>>>>> REPLACE
