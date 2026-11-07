import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin

    def try_greedy_build(order):
        S = []
        n = 0
        dp = [0] * 61
        dp[0] = 1
        all_sums = 1
        
        for x in order:
            if x in S:
                continue
            
            valid = True
            for k in range(1, n + 2):
                if k - 1 >= len(dp):
                    continue
                ns = dp[k-1] << x
                if ns & (all_sums & ~dp[k]):
                    valid = False
                    break
            
            if valid:
                S.append(x)
                n += 1
                old_dp = dp[:]
                for k in range(n, 0, -1):
                    dp[k] = old_dp[k] | (old_dp[k-1] << x)
                all_sums |= (all_sums << x)

        return S

    best_set = []
    
    # Generate diverse orders
    orders = []
    
    # 1. Sorted ascending and descending
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Alternating small/large (multiple variants)
    for split in [64, 96, 128, 160, 192]:
        small = list(range(1, split + 1))
        large = list(range(256, split, -1))
        alt = []
        for i in range(max(len(small), len(large))):
            if i < len(small):
                alt.append(small[i])
            if i < len(large):
                alt.append(large[i])
        orders.append(alt)
    
    # 3. Interleaved ranges (block interleaving)
    for block_size in [16, 32, 64]:
        o = []
        for start in range(0, 256, block_size * 2):
            for i in range(block_size):
                if start + i <= 256:
                    o.append(start + i + 1)
                if start + block_size + i <= 256:
                    o.append(start + block_size + i + 1)
        for v in range(1, 257):
            if v not in o:
                o.append(v)
        orders.append(o)
    
    # 4. Random with different seeds
    rng = random.Random(42)
    for trial in range(150):
        if time.time() > deadline - 5:
            break
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
    
    # 5. Spaced orders
    for step in [2, 3, 4, 5, 8, 16]:
        o = []
        for i in range(1, 257, step):
            o.append(i)
        for v in range(1, 257):
            if v not in o:
                o.append(v)
        orders.append(o)
        
    # 6. Descending spaced
    for step in [2, 3, 4, 5, 8, 16]:
        o = []
        for i in range(256, 0, -step):
            o.append(i)
        for v in range(256, 0, -1):
            if v not in o:
                o.append(v)
        orders.append(o)
    
    # 7. Power-of-2 biased order
    pow2_order = []
    for i in range(1, 9):
        pow2_order.append(2**i)
    for v in range(1, 257):
        if v not in pow2_order:
            pow2_order.append(v)
    orders.append(pow2_order)
    
    # 8. Fibonacci-like order
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    fib_order = list(fib)
    for v in range(1, 257):
        if v not in fib_order:
            fib_order.append(v)
    orders.append(fib_order)
    
    # 9. Large-first biased random
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        o = list(range(1, 257))
        weighted = []
        for v in o:
            weight = v / 256.0
            if rng.random() < weight:
                weighted.append(v)
        for v in o:
            if v not in weighted:
                weighted.append(v)
        orders.append(weighted)
    
    # 10. Small-first biased random
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        o = list(range(1, 257))
        weighted = []
        for v in o:
            weight = (257 - v) / 256.0
            if rng.random() < weight:
                weighted.append(v)
        for v in o:
            if v not in weighted:
                weighted.append(v)
        orders.append(weighted)

    for o in orders:
        if time.time() > deadline - 2:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            if len(S) >= 15:
                deadline = start_time + 55

    return [int(x) for x in best_set]
