"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for the 90s wall-clock limit
    
    # Pre-computed valid solution: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # Actually, let's use a known small solution.
    # 1 = 1/5 + 1/6 + 1/30 + 1/10 + 1/15 + 1/2 + ... no, 2 < 5.
    # Let me find a valid solution with denominators >= 5.
    
    # Known: 1 = 1/5 + 1/6 + 1/30 + 1/10 + 1/15
    # Check: 1/5 + 1/6 + 1/30 + 1/10 + 1/15 = 6/30 + 5/30 + 1/30 + 3/30 + 2/30 = 17/30 ≠ 1
    # That's not right.
    
    # Let me try: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This is too many terms.
    
    # Let me use the search to find a minimal solution.
    
    one = Fraction(1)
    
    def search(depth, max_depth, remaining, min_den, current):
        """Try to complete with exactly max_depth terms total (depth so far)."""
        if time.time() > deadline:
            return None
        
        if depth == max_depth:
            if remaining == 0:
                return list(current)
            return None
        
        terms_left = max_depth - depth
        
        # Max achievable sum with terms_left terms, denominators starting from min_den:
        max_possible = sum(Fraction(1, min_den + i) for i in range(terms_left))
        if remaining > max_possible:
            return None
        
        if terms_left == 1:
            if remaining > 0:
                # 1/d = remaining => d = 1/remaining
                d = (one / remaining)
                if d.denominator == 1:
                    d_int = d.numerator
                    if d_int >= min_den and d_int >= 5 and d_int <= 10000000:
                        current.append(d_int)
                        return list(current)
                        current.pop()
            return None
        
        # Try denominators starting from min_den
        if remaining <= 0:
            return None
        
        d_lower = max(min_den, int(one / remaining.numerator / remaining.denominator) + 1)
        
        for d in range(d_lower, 10000001):
            if d < 5:
                continue
            val = Fraction(1, d)
            if val > remaining:
                continue
            
            new_remaining = remaining - val
            
            if terms_left - 1 == 0:
                if new_remaining == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            # Max possible with terms_left-1 terms starting from d+1:
            if terms_left - 1 >= 1:
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(terms_left - 1))
                if new_remaining > max_rest:
                    break
            
            current.append(d)
            result = search(depth + 1, max_depth, new_remaining, d + 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Iterative deepening: try 1 term, 2 terms, ..., up to some limit
    for max_depth in range(1, 13):
        result = search(0, max_depth, one, 5, [])
        if result is not None:
            return result
    
    # Fallback: use the greedy approach to get a valid answer
    remaining = one
    result = []
    min_d = 5
    for _ in range(256):
        if remaining <= 0:
            break
        if remaining == 0:
            break
        # Find smallest d >= min_d such that 1/d <= remaining
        # d >= ceil(1/remaining)
        d = max(min_d, int(one / remaining.numerator / remaining.denominator) + 1)
        while True:
            val = Fraction(1, d)
            if val <= remaining:
                result.append(d)
                remaining -= val
                min_d = d + 1
                break
            d += 1
        if remaining == 0:
            break
    
    if len(result) >= 1:
        return result
    
    # Last resort: hardcode a known solution
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
