"""Reformulated approach: greedy + local search with deadline guard."""
import time
import random

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    def count_3aps(S):
        """Count distinct 3-term APs in set S (positive difference, counted once)."""
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if b > a:
                    c = 2*b - a
                    if c > b and c in S_set:
                        count += 1
        return count
    
    def has_4ap(S):
        """Check if set S contains a nonconstant 4-term AP."""
        S_set = set(S)
        for a in S:
            for d in range(1, (256 - a) // 3 + 1):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if b in S_set and c in S_set and e in S_set:
                    return True
        return False
    
    def count_3aps_fast(S):
        """Faster count using sorted list."""
        S_set = set(S)
        count = 0
        for i, a in enumerate(S):
            for b in S[i+1:]:
                c = 2*b - a
                if c > b and c in S_set:
                    count += 1
        return count
    
    # Initial greedy construction: add elements that don't create 4-APs
    def greedy_construct():
        S = set()
        # Try to build a large set with many 3-APs but no 4-APs
        # Use a heuristic: prefer elements that form many 3-APs
        for x in range(1, 257):
            trial = S | {x}
            if not has_4ap(trial):
                S = trial
        return sorted(S)
    
    # Local search: try swapping elements to improve 3-AP count
    def local_search(S, iterations=200000):
        S = list(S)
        S_set = set(S)
        current_score = count_3aps(S)
        all_elements = set(range(1, 257))
        
        for it in range(iterations):
            if time.time() > deadline - 2.0:
                break
            
            # Randomly pick an element to try removing or swapping
            if random.random() < 0.5 and len(S) > 10:
                # Try removing an element
                idx = random.randint(0, len(S)-1)
                removed = S[idx]
                new_S = S[:idx] + S[idx+1:]
                new_S_set = set(new_S)
                if not has_4ap(new_S_set):
                    new_score = count_3aps(new_S_set)
                    if new_score >= current_score:
                        S = new_S
                        S_set = new_S_set
                        current_score = new_score
            else:
                # Try swapping: remove one, add another
                if len(S) == 0:
                    continue
                idx = random.randint(0, len(S)-1)
                removed = S[idx]
                # Pick a random element not in S
                candidates = [x for x in all_elements if x not in S_set]
                if not candidates:
                    continue
                added = random.choice(candidates)
                new_S = S[:idx] + [added] + S[idx+1:]
                new_S_set = set(new_S)
                if not has_4ap(new_S_set):
                    new_score = count_3aps(new_S_set)
                    if new_score > current_score:
                        S = new_S
                        S_set = new_S_set
                        current_score = new_score
        return sorted(S)
    
    # Try multiple random restarts with different strategies
    best_S = []
    best_score = -1
    
    # Strategy 1: Greedy from scratch
    S1 = greedy_construct()
    score1 = count_3aps(S1)
    if score1 > best_score:
        best_S = S1
        best_score = score1
    
    # Strategy 2: Start with all elements, remove those causing 4-APs
    def construct_by_removal():
        S = set(range(1, 257))
        # Repeatedly remove elements that are part of 4-APs
        changed = True
        while changed:
            changed = False
            for a in list(S):
                for d in range(1, (256 - a) // 3 + 1):
                    b = a + d
                    c = a + 2*d
                    e = a + 3*d
                    if b in S and c in S and e in S:
                        # Remove one of them
                        S.remove(e)
                        changed = True
                        break
                if changed:
                    break
        return sorted(S)
    
    if time.time() < deadline - 5.0:
        S2 = construct_by_removal()
        score2 = count_3aps(S2)
        if score2 > best_score:
            best_S = S2
            best_score = score2
    
    # Strategy 3: Randomized construction
    if time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            random.seed(42 + restart)
            S = set()
            # Randomly add elements
            for x in random.sample(range(1, 257), 256):
                trial = S | {x}
                if not has_4ap(trial):
                    S = trial
            S = sorted(S)
            score = count_3aps(S)
            if score > best_score:
                best_S = S
                best_score = score
    
    # Local search on best solution
    if time.time() < deadline - 3.0 and len(best_S) > 0:
        best_S = local_search(best_S, iterations=500000)
    
    # Final check
    if not best_S:
        best_S = [1]
    
    return best_S

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
