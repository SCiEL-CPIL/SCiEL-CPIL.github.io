import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    def cost(S):
        """
        Returns a tuple (is_valid, score).
        If valid, score is max(S).
        If invalid, score is a penalty.
        We want to minimize score.
        """
        if not S:
            return False, float('inf')
        
        S_sorted = sorted(S)
        n = len(S_sorted)
        
        # Check Sidon
        diffs = []
        is_sidon_flag = True
        for i in range(n):
            for j in range(i+1, n):
                d = S_sorted[j] - S_sorted[i]
                if d in diffs:
                    is_sidon_flag = False
                    # We can break early if we want, but counting duplicates gives a gradient
                else:
                    diffs.append(d)
        
        if not is_sidon_flag:
            # Penalty: number of duplicate diffs + max element
            # A strong penalty ensures we fix Sidon first
            return False, 10000 + len(diffs)
            
        # Check Coverage
        diff_set = set(diffs)
        missing = 0
        for d in range(1, 31):
            if d not in diff_set:
                missing += 1
                
        if missing > 0:
            # Penalty: missing diffs + max element
            return False, 5000 + missing * 10 + S_sorted[-1]
            
        # Valid
        return True, S_sorted[-1]

    # Initial Solution: Greedy Construction
    def greedy_init():
        S = [0]
        covered = set()
        # Try to build a small set quickly
        while len(S) < 10:
            found = False
            for d in range(1, 31):
                if d in covered: continue
                for s in S:
                    c = s + d
                    if c in S or c < 0: continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found: break
            if not found:
                # Fallback: add large number
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

    # Simulated Annealing
    def simulated_annealing(initial_S, time_limit):
        rng = random.Random(42)
        
        # Start with a valid-ish set, maybe expand it to cover all diffs first?
        # Actually, let's start with a set that is definitely Sidon, and try to cover diffs.
        # Or start with the greedy result.
        
        current_S = sorted(initial_S)
        # Ensure we have enough elements? A Sidon set of size k has k(k-1)/2 diffs.
        # We need 30 diffs. k=8 gives 28, k=9 gives 36. So we need at least 9 elements to cover 30 diffs in a Sidon set?
        # Wait, not all diffs need to be distinct if we only care about 1..30?
        # NO, "all positive differences distinct" is the Sidon definition.
        # So if we have 8 elements, we have 28 distinct positive differences. We cannot cover 30 distinct values 1..30.
        # So we MUST have at least 9 elements.
        # Max size 32.
        
        if len(current_S) < 9:
            # Add random large elements to reach size 9
            while len(current_S) < 9:
                c = max(current_S) + rng.randint(1, 100)
                new_S = current_S + [c]
                new_S.sort()
                if is_sidon(new_S):
                    current_S = new_S
                else:
                    c += 1
                    
        current_valid, current_cost = cost(current_S)
        if not current_valid:
            # If greedy failed to make it valid, we need a better init or just let SA fix it.
            # But SA with high penalties might struggle. Let's try to force validity.
            pass

        best_S = current_S[:]
        best_cost = current_cost
        best_valid = current_valid
        
        T = 1000.0
        cooling = 0.999
        min_T = 1.0
        
        while time.time() - start_time < deadline - 5:
            if T < min_T:
                break
                
            # Generate neighbor
            neighbor_S = current_S[:]
            move_type = rng.choice(['move', 'add', 'remove', 'shift'])
            
            if move_type == 'move' and len(neighbor_S) > 0:
                idx = rng.randint(0, len(neighbor_S) - 1)
                # Replace neighbor_S[idx] with a new value
                # Keep it within a reasonable range
                max_val = max(neighbor_S) if neighbor_S else 0
                new_val = rng.randint(1, max(100, max_val + 50))
                if new_val != neighbor_S[idx]:
                    neighbor_S[idx] = new_val
                    neighbor_S.sort()
                    # Check for duplicates
                    if len(neighbor_S) != len(set(neighbor_S)):
                        continue
            elif move_type == 'add' and len(neighbor_S) < 32:
                max_val = max(neighbor_S) if neighbor_S else 0
                new_val = rng.randint(1, max(100, max_val + 50))
                if new_val not in neighbor_S:
                    neighbor_S.append(new_val)
                    neighbor_S.sort()
            elif move_type == 'remove' and len(neighbor_S) > 9:
                idx = rng.randint(0, len(neighbor_S) - 1)
                neighbor_S.pop(idx)
            elif move_type == 'shift':
                # Shift a suffix
                if len(neighbor_S) > 1:
                    split = rng.randint(1, len(neighbor_S) - 1)
                    delta = rng.randint(-5, 5)
                    if delta != 0:
                        for i in range(split, len(neighbor_S)):
                            neighbor_S[i] += delta
                        # Ensure non-negative and unique
                        if any(x < 0 for x in neighbor_S) or len(neighbor_S) != len(set(neighbor_S)):
                            continue
                        neighbor_S.sort()
            
            if not neighbor_S:
                continue
                
            neighbor_valid, neighbor_cost = cost(neighbor_S)
            
            # Update best
            if neighbor_valid:
                if not best_valid or neighbor_cost < best_cost:
                    best_valid = True
                    best_cost = neighbor_cost
                    best_S = neighbor_S[:]
            
            # Acceptance
            if neighbor_cost < current_cost:
                current_S = neighbor_S
                current_valid, current_cost = neighbor_valid, neighbor_cost
            else:
                delta = neighbor_cost - current_cost
                if rng.random() < pow(2.71828, -delta / T):
                    current_S = neighbor_S
                    current_valid, current_cost = neighbor_valid, neighbor_cost
                    
            T *= cooling
            
        return best_S

    # Run SA with a few restarts if time permits
    best_S_overall = None
    best_cost_overall = float('inf')
    
    # Initial Greedy
    S_init = greedy_init()
    
    # First SA run
    S_res = simulated_annealing(S_init, deadline - 20)
    v, c = cost(S_res)
    if v and c < best_cost_overall:
        best_cost_overall = c
        best_S_overall = S_res
    
    # If we have time, do more restarts with perturbed initial states
    while time.time() - start_time < deadline - 10:
        # Perturb best_S_overall
        if best_S_overall:
            S_pert = best_S_overall[:]
            rng = random.Random(int(time.time()*1000))
            for _ in range(5):
                if not S_pert: break
                idx = rng.randint(0, len(S_pert)-1)
                S_pert[idx] += rng.randint(-20, 20)
            S_pert.sort()
            S_pert = list(set(S_pert))
            
            S_res = simulated_annealing(S_pert, deadline - 10)
            v, c = cost(S_res)
            if v and c < best_cost_overall:
                best_cost_overall = c
                best_S_overall = S_res
        else:
            break

    if best_S_overall:
        return best_S_overall
    else:
        # Fallback: Return the greedy result even if invalid? No, must be valid.
        # If we didn't find a valid one, we should try harder or return a known small valid set if we have one hardcoded.
        # But greedy_init should produce a Sidon set. It might not cover 1..30.
        # Let's force a valid set construction as a last resort.
        # A known Sidon set covering 1..30?
        # If we fail, return a set that is at least Sidon and includes 0.
        # The task requires covering 1..30. If we can't, we score 0.
        # But greedy_init + SA should find one.
        return S_init

