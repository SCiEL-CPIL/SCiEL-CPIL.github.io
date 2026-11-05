"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def can_add(selected, x):
        """Check if x can be added without creating a 4-term AP."""
        # x as 1st term: need x+d, x+2d, x+3d in selected
        for d in range(1, (N - x) // 3 + 1):
            if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                return False
        # x as 2nd term: need x-d, x+d, x+2d in selected
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                return False
        # x as 3rd term: need x-2d, x-d, x+d in selected
        for d in range(1, min(x-2, N-x) + 1):
            if (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                return False
        # x as 4th term: need x-3d, x-2d, x-d in selected
        for d in range(1, x//3 + 1):
            if (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                return False
        return True
    
    def greedy_add(selected, order):
        for x in order:
            if x not in selected and can_add(selected, x):
                selected.add(x)
        return selected
    
    def swap_local_search(selected, max_rounds=50):
        """Try removing one and adding another to increase size."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time_left() > 10:
            improved = False
            rounds += 1
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rx in sel_list:
                if time_left() < 10:
                    break
                trial = selected - {rx}
                # Try adding candidates
                candidates = [y for y in range(1, N+1) if y not in trial]
                random.shuffle(candidates)
                for ay in candidates[:150]:
                    if can_add(trial, ay):
                        trial.add(ay)
                        if len(trial) > len(selected):
                            selected = trial
                            improved = True
                            break
                if improved:
                    break
        return selected
    
    best_set = set()
    
    # Natural order
    if time_left() > 3:
        s = greedy_add(set(), list(range(1, N+1)))
        if len(s) > len(best_set):
            best_set = s
    
    # Reverse order
    if time_left() > 3:
        s = greedy_add(set(), list(range(N, 0, -1)))
        if len(s) > len(best_set):
            best_set = s
    
    # Random orderings
    random.seed(12345)
    for _ in range(80):
        if time_left() < 3:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        s = greedy_add(set(), order)
        if len(s) > len(best_set):
            best_set = s
    
    # AP-density ordering
    if time_left() > 3:
        ap_count = [0] * (N+1)
        for a in range(1, N+1):
            for d in range(1, (N-a)//3+1):
                a4 = a + 3*d
                if a4 <= N:
                    for x in [a, a+d, a+2*d, a4]:
                        ap_count[x] += 1
        order = sorted(range(1, N+1), key=lambda x: ap_count[x])
        s = greedy_add(set(), order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search on best
    if time_left() > 15:
        best_set = swap_local_search(best_set, max_rounds=200)
    
    # More restarts with local search
    for _ in range(30):
        if time_left() < 20:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        s = greedy_add(set(), order)
        if time_left() < 15:
            break
        s = swap_local_search(s, max_rounds=50)
        if len(s) > len(best_set):
            best_set = s
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
