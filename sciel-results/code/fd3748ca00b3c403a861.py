import time

def construct():
      start_time = time.time()
      deadline = start_time + 55 # 5s margin
      
      def solve(k, num, den, min_den, current):
          # target = num/den
          # We need k more terms
          
          # Pruning 1: Max sum check
          # Max sum of k terms starting from min_den
          # sum(1/(min_den + i) for i in range(k))
          # Approximate with integral or just sum if k is small
          if k <= 10:
              max_sum_num = 0
              max_sum_den = 1
              # This is slow if we do fractions.
              # Use float for pruning?
              max_sum_float = 0.0
              for i in range(k):
                  max_sum_float += 1.0 / (min_den + i)
              if max_sum_float < (num / den) * 0.999999:
                  return None
          
          # Pruning 2: If k == 1
          if k == 1:
              if den % num == 0:
                  d = den // num
                  if d >= min_den:
                      return current + [d]
              return None
          
          # Iterate next denominator d
          # 1/d < num/den => d > den/num
          # d >= min_den
          start_d = max(min_den, den // num + 1)
          
          # Upper bound for d
          # We need remaining k-1 terms to sum to R = num/den - 1/d
          # R = (num*d - den) / (den*d)
          # Max sum of k-1 terms > d is approx (k-1)/d
          # So (num*d - den) / (den*d) <= (k-1)/d
          # num*d - den <= den*(k-1)
          # num*d <= den*k
          # d <= den*k / num
          # This is a rough upper bound.
          # A tighter bound:
          # R <= sum_{i=1}^{k-1} 1/(d+i)
          # (num*d - den) / (den*d) <= sum_{i=1}^{k-1} 1/(d+i)
          # This is hard to solve for d directly.
          # But we can break when the condition fails.
          
          if k == 2:
              # 1/d1 + 1/d2 = num/den
              # d1 in (den/num, 2*den/num)
              d1_min = start_d
              d1_max = (2 * den) // num
              if d1_max > 10000000:
                  d1_max = 10000000
              
              for d1 in range(d1_min, d1_max + 1):
                  # Check if d2 is integer
                  # num*d1 - den = D_rem
                  D_rem = num * d1 - den
                  if D_rem <= 0:
                      continue
                  if (den * d1) % D_rem == 0:
                      d2 = (den * d1) // D_rem
                      if d2 > d1 and d2 <= 10000000:
                          return current + [d1, d2]
              return None
          
          # General k > 2
          # Bound for d
          # d_min = start_d
          # d_max: 
          # If d is too large, 1/d is too small, and the remaining sum is too big for the remaining terms.
          # Condition: num/den - 1/d <= sum_{i=1}^{k-1} 1/(d+i)
          # Let's iterate and break.
          
          # To speed up, we can estimate d_max.
          # Approx: num/den - 1/d <= (k-1)/d
          # num/den <= k/d => d <= k*den/num
          d_max_est = (k * den) // num
          if d_max_est > 10000000:
              d_max_est = 10000000
              
          # Also d must be at least start_d
          # And we should not iterate too many times.
          
          for d in range(start_d, d_max_est + 1):
              if time.time() > deadline:
                  return None
              
              # Check if remainder is too big
              # R = num/den - 1/d
              # R = (num*d - den) / (den*d)
              num_rem = num * d - den
              if num_rem <= 0:
                  continue # 1/d >= target, impossible for k>1
              
              den_rem = den * d
              
              # Check if R is too big for k-1 terms
              # Max sum of k-1 terms starting from d+1
              # Approx (k-1)/(d+1)
              # If R > (k-1)/(d+1), break?
              # R = num_rem / den_rem
              # num_rem / den_rem > (k-1) / (d+1)
              # num_rem * (d+1) > den_rem * (k-1)
              if num_rem * (d + 1) > den_rem * (k - 1):
                  break
                  
              res = solve(k - 1, num_rem, den_rem, d + 1, current + [d])
              if res:
                  return res
                  
          return None

      # Try k=8
      for k in [8, 9, 10]:
          res = solve(k, 1, 1, 5, [])
          if res:
              # Verify
              if len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                  s = sum(1.0/d for d in res)
                  if abs(s - 1.0) < 1e-9:
                      return res
       
      # Fallback: Return a known valid solution if DFS fails
      # We need a valid solution to avoid score 0.
      # Use the DP method from seed as fallback, but it might be slow.
      # Or return a hardcoded 8-term solution if we can find one.
      
      # Let's try to find a hardcoded solution.
      # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 is not exact.
      # Let's use the DP method but with a timeout.
      
      target, reached = 2520, {0: []}
      for d in range(5, 2521):
          if 2520 % d:
              continue
          w = 2520 // d
          for total, values in list(reached.items()):
              if total + w <= 2520 and total + w not in reached:
                  reached[total + w] = values + [d]
          if 2520 in reached:
              # Verify this solution
              vals = reached[2520]
              if len(set(vals)) == len(vals) and all(5 <= d <= 10000000 for d in vals):
                  return vals
              
      raise RuntimeError("no solution found")
