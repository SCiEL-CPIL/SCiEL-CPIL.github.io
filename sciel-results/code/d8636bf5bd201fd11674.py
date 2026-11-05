   def construct():
       import time
       import random
       import math
       
       start_time = time.time()
       deadline = start_time + 85
       
       # LCM
       def lcm(a, b):
           return abs(a*b) // math.gcd(a, b)
       
       L = 1
       for i in range(1, 31):
           L = lcm(L, i)
           
       weights = [L // i for i in range(1, 31)]
       
       # Seed
       seed_c = [0]*28 + [1, -1]
       seed_s = sum(seed_c[i] * weights[i] for i in range(30))
       best_abs = abs(seed_s)
       best_c = seed_c[:]
       
       def evaluate(c):
           return sum(c[i] * weights[i] for i in range(30))
       
       def sa_run(c_init, t_start, t_end, iters):
           c = c_init[:]
           s = evaluate(c)
           cur_abs = abs(s)
           best_c_local = c[:]
           best_abs_local = cur_abs
           
           T = t_start
           cooling = (t_end / t_start) ** (1.0 / iters)
           
           # Pre-compute choices? No, random is fine.
           
           for i in range(iters):
               if time.time() > deadline - 2:
                   break
               
               idx = random.randint(0, 29)
               old_val = c[idx]
               # Pick new val
               # 1/3 chance 0, 1/3 chance 1, 1/3 chance -1
               # If old_val is 0, new is 1 or -1
               # If old_val is 1, new is 0 or -1
               # If old_val is -1, new is 0 or 1
               
               if old_val == 0:
                   new_val = 1 if random.random() < 0.5 else -1
               elif old_val == 1:
                   new_val = 0 if random.random() < 0.5 else -1
               else: # -1
                   new_val = 0 if random.random() < 0.5 else 1
                   
               delta = (new_val - old_val) * weights[idx]
               new_s = s + delta
               
               if new_s == 0:
                   # Reject
                   continue
                   
               new_abs = abs(new_s)
               
               if new_abs < cur_abs:
                   c[idx] = new_val
                   s = new_s
                   cur_abs = new_abs
                   if new_abs < best_abs_local:
                       best_abs_local = new_abs
                       best_c_local = c[:]
               else:
                   # Accept with prob
                   # Use a fixed T for simplicity in inner loop?
                   # T changes slowly.
                   # Approximation:
                   if random.random() < math.exp(-(new_abs - cur_abs) / T):
                       c[idx] = new_val
                       s = new_s
                       cur_abs = new_abs
                       
               T *= cooling
               
           return best_c_local, best_abs_local
       
       # Run restarts
       # 1. Start from seed
       # 2. Start from random
       
       # Determine iterations based on time
       # Estimate 1M iters/sec
       
       # Run 1 long SA from seed
       time_left = deadline - time.time()
       iters = int(time_left * 0.5 * 1000000) # 50% of time
       bc, ba = sa_run(seed_c, 100000, 1, iters)
       if ba < best_abs:
           best_abs = ba
           best_c = bc
           
       # Run random restarts with remaining time
       while time.time() < deadline - 5:
           c_init = [random.choice([-1, 0, 1]) for _ in range(30)]
           while sum(c_init[i] * weights[i] for i in range(30)) == 0:
               c_init[random.randint(0,29)] = random.choice([-1, 0, 1])
               
           time_left = deadline - time.time() - 5
           if time_left <= 0:
               break
           
           iters = int(time_left * 0.8 * 1000000)
           if iters < 1000:
               break
               
           bc, ba = sa_run(c_init, 50000, 1, iters)
           if ba < best_abs:
               best_abs = ba
               best_c = bc
               
       return best_c
   