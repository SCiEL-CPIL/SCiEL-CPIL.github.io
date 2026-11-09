def construct():
        import time
        start = time.time()
        deadline = start + 85
        
        N = 1000000
        WINDOW = 24
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        MAX_VAL = N + WINDOW

        # Precompute sp
        lp = [0] * (MAX_VAL + 1)
        for p in PRIMES:
            for m in range(p, MAX_VAL + 1, p):
                if lp[m] == 0:
                    lp[m] = p

        sp = [1] * (MAX_VAL + 1)
        for m in range(2, MAX_VAL + 1):
            p = lp[m]
            if p != 0:
                sp[m] = sp[m // p] * p
            else:
                sp[m] = 1
        
        best_n = 1000
        best_distinct = WINDOW + 1
        
        freq = {}
        # Initialize window for n=1000: sp[1001] ... sp[1024]
        current_distinct = 0
        for i in range(1001, 1001 + WINDOW):
            val = sp[i]
            if val in freq:
                freq[val] += 1
            else:
                freq[val] = 1
                current_distinct += 1
                
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = 1000
            
        for n in range(1001, N + 1):
            if time.time() > deadline:
                break
                
            # Remove sp[n]
            old_val = sp[n]
            freq[old_val] -= 1
            if freq[old_val] == 0:
                del freq[old_val]
                current_distinct -= 1
                
            # Add sp[n+WINDOW]
            new_val = sp[n + WINDOW]
            if new_val in freq:
                freq[new_val] += 1
            else:
                freq[new_val] = 1
                current_distinct += 1
                
            if current_distinct < best_distinct:
                best_distinct = current_distinct
                best_n = n
                
        return best_n
