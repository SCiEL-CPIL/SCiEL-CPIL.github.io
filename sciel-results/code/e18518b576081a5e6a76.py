from fractions import Fraction
import time

_DEADLINE = None
_BEST = None

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 50.0
    _BEST = None

def _subset_sum_exists(target, elements):
    if not elements:
        return False
    total = sum(Fraction(1, y) for y in elements)
    if total < target:
        return False
    elems = sorted(elements)
    n = len(elems)
    
    def dfs(idx, acc):
        if acc == target:
            return True
        if idx == n or acc > target:
            return False
        remaining_sum = sum(Fraction(1, elems[i]) for i in range(idx, n))
        if acc + remaining_sum < target:
            return False
        if dfs(idx + 1, acc + Fraction(1, elems[idx])):
            return True
        if dfs(idx + 1, acc):
            return True
        return False
    
    return dfs(0, Fraction(0))

def _can_add(x, selected):
    return not _subset_sum_exists(Fraction(1, x), selected)

def construct():
    _init()
    
    greedy_best = []
    for x in range(24, 0, -1):
        if _can_add(x, greedy_best):
            greedy_best.append(x)
    
    best = list(greedy_best)
    
    def dfs(x, selected):
        nonlocal best
        if time.time() > _DEADLINE - 5:
            return
        if x == 0:
            if len(selected) > len(best):
                best = list(selected)
            return
        if len(selected) + x <= len(best):
            return
        
        if _can_add(x, selected):
            selected.append(x)
            dfs(x - 1, selected)
            selected.pop()
        
        dfs(x - 1, selected)
    
    dfs(24, [])
    
    if not best:
        best = [24]
    
    _BEST = best
    return list(_BEST)
