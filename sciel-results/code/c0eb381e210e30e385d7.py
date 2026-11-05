import time
import random
import sys

_DEADLINE = None
_TIME_LIMIT = 145.0

def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())

def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Lower bound for max entry
    lb = n_rows * k * (k + 1) // 2 + 1
    
    def gen_ruler(L, k, max_tries=2000):
        for _ in range(max_tries):
            points = [0] + random.sample(range(1, L + 1), k)
            points.sort()
            
            diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = points[j] - points[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                return points, diffs
        return None, None

    def solve(L, max_backtracks=5000):
        used_diffs = set()
        rows = []
        
        def backtrack(idx, backtracks_used):
            if idx == n_rows:
                return rows[:]
            
            if backtracks_used > max_backtracks:
                return None
                
            for _ in range(20):
                ruler, diffs = gen_ruler(L, k)
                if ruler is None:
                    continue
                
                if diffs.isdisjoint(used_diffs):
                    rows.append(ruler)
                    used_diffs.update(diffs)
                    
                    result = backtrack(idx + 1, backtracks_used + 1)
                    if result is not None:
                        return result
                    
                    rows.pop()
                    used_diffs.difference_update(diffs)
                    
            return None

        return backtrack(0, 0)

    def _validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    current_L = lb
    while _time_left() > 10.0:
        result = solve(current_L, max_backtracks=2000)
        if result is not None:
            if _validate(result, n_rows, k):
                return result
        
        current_L += 1
        
        if _time_left() < 30.0:
             for _ in range(5):
                 if _time_left() < 5.0: break
                 big_L = random.randint(150, 250)
                 result = solve(big_L, max_backtracks=5000)
                 if result is not None and _validate(result, n_rows, k):
                     return result

    result = solve(300, max_backtracks=10000)
    if result and _validate(result, n_rows, k):
        return result

    def random_valid_set():
        used = set()
        rows = []
        for _ in range(n_rows):
            while True:
                ruler, diffs = gen_ruler(500, k)
                if ruler and diffs.isdisjoint(used):
                    rows.append(ruler)
                    used.update(diffs)
                    break
        return rows

    return random_valid_set()
