import time
import random

L = 5354228880
N = 24
ALL = list(range(1, 25))
RECIP = [0] + [L // i for i in range(1, 25)]

def precompute_conflicts():
    conflicts = {}
    for x in range(1, 25):
        target = L // x
        candidates = [i for i in range(1, 25) if i != x]
        solutions = []
        
        def backtrack(start, current_sum, current_set):
            if current_sum == target:
                if current_set:
                    solutions.append(frozenset(current_set))
                return
            if current_sum > target:
                return
            
            for i in range(start, len(candidates)):
                val = candidates[i]
                w = L // val
                if current_sum + w > target:
                    continue
                
                current_set.append(val)
                backtrack(i + 1, current_sum + w, current_set)
                current_set.pop()
        
        backtrack(0, 0, [])
        conflicts[x] = solutions
        
    return conflicts

CONFLICTS = precompute_conflicts()

def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        for T in CONFLICTS[x]:
            if T.issubset(S_set):
                return False
    return True


def construct():
    start = time.time()
    deadline = start + 85.0

    def now():
        return time.time()

    best_set = None
    best_size = 0

    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    def local_search(S, max_iters=200, seed_offset=0):
        S = set(S)
        S_list = list(S)
        rng = random.Random(seed_offset + len(S) * 1000)
        iters = 0
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            missing = [x for x in ALL if x not in S]
            if missing:
                shuffled_missing = missing[:]
                rng.shuffle(shuffled_missing)
                for x in shuffled_missing:
                    if now() > deadline:
                        break
                    trial = list(S) + [x]
                    if is_valid(trial):
                        S.add(x)
                        S_list = list(S)
                        improved = True
                        break
            if improved:
                continue

            if len(S) > 0:
                missing = [x for x in ALL if x not in S]
                if missing:
                    a = rng.choice(list(S))
                    shuffled_missing = missing[:]
                    rng.shuffle(shuffled_missing)
                    for b in shuffled_missing:
                        if now() > deadline:
                            break
                        new_set = (S - {a}) | {b}
                        if is_valid(list(new_set)):
                            S = new_set
                            S_list = list(S)
                            improved = True
                            break

            if not improved:
                break

        return S_list

    if best_set is not None:
        ls = local_search(best_set, max_iters=150, seed_offset=0)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    for restart in range(100):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        
        S = set(rand_set)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx = None
            for i, x in enumerate(S_list):
                for T in CONFLICTS[x]:
                    if T.issubset(S):
                        viol_idx = i
                        break
                if viol_idx is not None:
                    break
            if viol_idx is None:
                break
            S.remove(S_list[viol_idx])
        
        S_list = list(S)
        missing = [x for x in ALL if x not in S]
        for x in missing:
            if now() > deadline:
                break
            trial = S_list + [x]
            if is_valid(trial):
                S.add(x)
                S_list = list(S)
        
        ls = local_search(list(S), max_iters=100, seed_offset=restart)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    if best_set is not None and now() < deadline - 5.0:
        ls_final = local_search(best_set, max_iters=200, seed_offset=9999)
        if len(ls_final) > best_size:
            best_size = len(ls_final)
            best_set = ls_final[:]

    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
