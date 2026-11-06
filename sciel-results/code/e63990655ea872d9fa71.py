"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precomputed known Egyptian fraction for 1 with denominators >= 5.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/20 + 1/42 + 1/70 + 1/140 + 1/210 + 1/420
    # Let's verify: 
    # Actually, a well-known short representation:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70
    # Let's use a verified one. 
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/20 + 1/42 + 1/70 + 1/140 + 1/210 + 1/420
    # Sum = (840 + 700 + 600 + 420 + 210 + 100 + 60 + 30 + 20 + 10) / 420 = 2991 / 420 = 7.12... No.
    
    # Let's use the standard greedy algorithm result or a known identity.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70
    # Let's check the sum of reciprocals of divisors of 420 greater than 4:
    # Divisors of 420: 1, 2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 28, 30, 35, 42, 60, 70, 84, 105, 140, 210, 420
    # Sum of reciprocals of ALL divisors of 420 is sigma(420)/420 = (1+2+3+4+5+6+7+10+12+14+15+20+21+28+30+35+42+60+70+84+105+140+210+420)/420
    # Sum = 1616. 1616/420 = 3.84...
    
    # Let's just use a hardcoded valid solution.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70
    # Let's verify this specific set:
    # LCM(5,6,7,10,15,21,30,35,42,70) = 210
    # 210/5 = 42
    # 210/6 = 35
    # 210/7 = 30
    # 210/10 = 21
    # 210/15 = 14
    # 210/21 = 10
    # 210/30 = 7
    # 210/35 = 6
    # 210/42 = 5
    # 210/70 = 3
    # Sum = 42+35+30+21+14+10+7+6+5+3 = 173. 173 != 210.
    
    # How about: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/20 + 1/42 + 1/70 + 1/140 + 1/210 + 1/420
    # LCM = 420
    # 84, 70, 60, 42, 21, 10, 6, 3, 2, 1. Sum = 299. No.
    
    # Let's use a known short one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/15 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
    # This is getting complicated. Let's just return a hardcoded valid list.
    # From literature, a short Egyptian fraction for 1 with denominators > 4:
    # [5, 6, 7, 10, 15, 21, 30, 35, 42, 70, 105, 210]
    # Sum of reciprocals:
    # LCM = 210
    # 42, 35, 30, 21, 14, 10, 7, 6, 5, 3, 2, 1
    # Sum = 42+35+30+21+14+10+7+6+5+3+2+1 = 176. No.
    
    # Actually, the simplest is just to return the original DP result if it's fast, or a hardcoded valid one.
    # The original DP finds a solution quickly. Let's just add a time guard and keep the DP.
    
    start_time = time.time()
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if time.time() - start_time > 50:
            if target in reached:
                return reached[target]
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    if target in reached:
        return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
