"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Hardcoded verified solution: 12 terms
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
    # Let me verify: LCM of these is 210
    # 210/5=42, 210/6=35, 210/7=30, 210/10=21, 210/15=14, 210/21=10, 
    # 210/30=7, 210/35=6, 210/42=5, 210/70=3, 210/105=2, 210/210=1
    # Sum = 42+35+30+21+14+10+7+6+5+3+2+1 = 176. Not 210. So this is wrong.
    
    # Let me find a correct solution. 
    # 1 = 1/2 + 1/3 + 1/6
    # Need to express 1/2, 1/3, 1/6 with denominators >= 5
    # 1/2 = 1/5 + 3/10; 3/10 = 1/4 + 1/20 (4<5, bad)
    # 3/10 = 1/6 + 1/15 = 5/30 + 2/30 = 7/30 ≠ 3/10. Bad.
    # 3/10 = 1/7 + 11/70; 11/70 = 1/8 + 5/56... 
    # 11/70 - 1/8 = 44/280 - 35/280 = 9/280. 9/280 = 1/32 + 1/... 
    # This is getting complex. Let me use a search.
    
    # Better approach: use the identity 1/n = 1/(n+1) + 1/(n(n+1))
    # 1/2 = 1/3 + 1/6
    # 1/3 = 1/4 + 1/12 (4<5, bad)
    # 1/4 = 1/5 + 1/20
    # 1/12 = 1/13 + 1/156
    # So 1/3 = 1/5 + 1/20 + 1/13 + 1/156
    # 1/6 = 1/7 + 1/42
    # 1/2 = 1/3 + 1/6 = 1/5 + 1/20 + 1/13 + 1/156 + 1/7 + 1/42
    # 1 = 1/2 + 1/3 + 1/6 = (above) + 1/3 + 1/6
    # But 1/3 = 1/5+1/20+1/13+1/156, so we get duplicates.
    
    # Let me try a different decomposition.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # This is getting too long. Let me use a proper search.
    
    # Use DFS with memoization to find minimal solution
    # State: (remaining_fraction, min_denom, used_set)
    # But used_set makes it hard to memoize. Instead, use a greedy + backtracking approach.
    
    # Actually, let me just try known solutions.
    # 1 = 1/2 + 1/3 + 1/6
    # Replace 1/2: 1/2 = 1/5 + 1/10 + 1/20 + 1/20... no, need distinct.
    # 1/2 = 1/5 + 3/10; 3/10 = 1/4 + 1/20 (bad); 3/10 = 1/6 + 1/15 (check: 5/30+2/30=7/30≠3/10)
    # 3/10 = 1/7 + 11/70; 11/70 = 1/8 + 9/280; 9/280 = 1/32 + 1/... 
    # 9/280 - 1/32 = 72/2240 - 70/2240 = 2/2240 = 1/1120
    # So 3/10 = 1/7 + 1/8 + 1/32 + 1/1120
    # 1/2 = 1/5 + 1/7 + 1/8 + 1/32 + 1/1120
    # 1/3 = 1/4 + 1/12 (bad); 1/3 = 1/5 + 2/15; 2/15 = 1/8 + 1/60 (check: 15/120+2/120=17/120≠2/15)
    # 2/15 = 1/9 + 1/45 (check: 5/45+1/45=6/45=2/15 ✓)
    # So 1/3 = 1/5 + 1/9 + 1/45
    # 1/6 = 1/7 + 1/42
    # 1 = 1/2 + 1/3 + 1/6 = (1/5+1/7+1/8+1/32+1/1120) + (1/5+1/9+1/45) + (1/7+1/42)
    # Duplicates: 1/5 appears twice, 1/7 appears twice. Bad.
    
    # Let me try yet another approach. Use the fact that:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520
    # This is 16 terms. Let me verify with LCM = 2520:
    # 2520/5=504, /6=420, /7=360, /10=252, /15=168, /21=120, /30=84, /35=72, /42=60, /70=36, /105=24, /210=12, /420=6, /840=3, /1260=2, /2520=1
    # Sum = 504+420+360+252+168+120+84+72+60+36+24+12+6+3+2+1 = 2124 ≠ 2520. Bad.
    
    # I'll use a computational search. Let me implement a proper DFS.
    
    best_sol = None
    
    def dfs(remaining, min_d, used, current, depth):
        nonlocal best_sol
        if time.time() > deadline:
            return
        if remaining == 0:
            if best_sol is None or len(current) < len(best_sol):
                best_sol = current[:]
            return
        if remaining < 0:
            return
        if depth > 12:  # limit depth
            return
        if best_sol is not None and depth >= len(best_sol):
            return
        
        # Try denominators from min_d upward
        # Upper bound: 1/d <= remaining, so d >= 1/remaining
        # But we also need d >= min_d and d not in used
        # For efficiency, try a limited range
        max_d = min(10000000, max(min_d, int(1 / remaining) + 100) if remaining > 0 else 10000000)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return
            if d in used:
                continue
            term = Fraction(1, d)
            if term > remaining:
                continue
            new_remaining = remaining - term
            if new_remaining < 0:
                continue
            used.add(d)
            current.append(d)
            dfs(new_remaining, d + 1, used, current, depth + 1)
            current.pop()
            used.remove(d)
            if best_sol is not None and len(current) + 1 >= len(best_sol):
                # Can't beat current best with this branch
                break
    
    dfs(Fraction(1, 1), 5, set(), [], 0)
    
    if best_sol is not None:
        # Verify
        s = sum(Fraction(1, d) for d in best_sol)
        if s == 1 and len(set(best_sol)) == len(best_sol) and all(5 <= d <= 10000000 for d in best_sol):
            return best_sol
    
    # Fallback: return a known valid long solution
    # Let me construct one by splitting 1 = 1/2 + 1/3 + 1/6
    # and expressing each with distinct denominators >= 5
    
    # 1/2 = 1/5 + 1/6 + 1/30? 6/30+5/30+1/30=12/30=2/5≠1/2. No.
    # 1/2 = 1/5 + 1/7 + 1/35? 7/35+5/35+1/35=13/35≠1/2. No.
    # 1/2 = 1/5 + 1/8 + 1/40? 8/40+5/40+1/40=14/40=7/20≠1/2. No.
    # 1/2 = 1/5 + 1/9 + 1/45? 9/45+5/45+1/45=15/45=1/3≠1/2. No.
    # 1/2 = 1/5 + 1/10 + 1/20 + 1/20... no distinct.
    # 1/2 - 1/5 = 3/10. 3/10 - 1/10 = 1/5. 1/5 - 1/20 = 3/20. 3/20 - 1/20 = 1/10. 1/10 - 1/10 = 0.
    # So 1/2 = 1/5 + 1/10 + 1/20 + 1/10... duplicate.
    # 3/10 = 1/4 + 1/20 (4<5). 3/10 = 1/6 + 1/15 = 7/30≠3/10. 
    # 3/10 = 1/7 + 11/70. 11/70 = 1/8 + 9/280. 9/280 = 1/32 + 1/1120.
    # So 1/2 = 1/5 + 1/7 + 1/8 + 1/32 + 1/1120
    
    # 1/3 = 1/5 + 1/9 + 1/45 (verified above)
    # 1/6 = 1/7 + 1/42
    
    # But 1/5 and 1/7 appear in both 1/2 and 1/3/1/6 decompositions.
    # Let me use different decompositions to avoid duplicates.
    
    # 1/2 = 1/5 + 1/7 + 1/8 + 1/32 + 1/1120
    # 1/3 = 1/6 + 1/12 + 1/36... no, 1/3 - 1/6 = 1/6. 1/6 - 1/12 = 1/12. 1/12 - 1/36 = 1/18. 1/18... 
    # This is getting too complex. Let me just return the DFS result or a hardcoded valid one.
    
    # After more thought, a valid 12-term solution is:
    # [5, 6, 7, 10, 15, 21, 30, 35, 42, 70, 105, 210]
    # But we showed sum = 176/210 ≠ 1.
    
    # Let me try: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520
    # Sum = 2124/2520 ≠ 1.
    
    # I'll just return the DFS result. If it's None, I'll construct a valid one.
    if best_sol is not None:
        return best_sol
    
    # Final fallback: use a very long but valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 + 1/5040 + 1/10080 + 1/20160 + 1/40320 + 1/80640 + 1/161280 + 1/322560 + 1/645120 + 1/1290240 + 1/2580480 + 1/5160960 + 1/10321920 + 1/20643840 + 1/41287680 + 1/82575360 + 1/165150720 + 1/330301440 + 1/660602880 + 1/1321205760 + 1/2642411520 + 1/5284823040 + 1/10569646080
    # This is way too long and likely not valid.
    
    # Let me just use a simple valid construction:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 + 1/5040 + 1/10080 + 1/20160 + 1/40320 + 1/80640 + 1/161280 + 1/322560 + 1/645120 + 1/1290240 + 1/2580480 + 1/5160960 + 1/10321920 + 1/20643840 + 1/41287680 + 1/82575360 + 1/165150720 + 1/330301440 + 1/660602880 + 1/1321205760 + 1/2642411520 + 1/5284823040 + 1/10569646080
    # This is not valid. Let me just raise an error if DFS fails.
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END
