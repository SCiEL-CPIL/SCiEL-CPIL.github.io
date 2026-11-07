     def construct():
         start = time.time()
         deadline = start + 50.0 # 10s buffer
         
         best_a, best_b = 11, 6 # Fallback
         best_num, best_den = 4, 12
         
         ranges = [10**6, 5*10**6, 10**7]
         
         for N in ranges:
             if time.time() > deadline - 5: # Need 5s for this stage?
                 # Estimate time for stage
                 # If N is 10^7, it takes ~20s.
                 # If N is 10^6, it takes ~1s.
                 pass
             
             result = find_best_pair(N, deadline - 2.0) # Leave 2s for next stage or return
             
             if result is not None:
                 a, b, num, den = result
                 if num * best_den < best_num * den:
                     best_a, best_b = a, b
                     best_num, best_den = num, den
                     
         return [int(best_a), int(best_b)]
     