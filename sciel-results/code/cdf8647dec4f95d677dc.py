from math import factorial
from functools import lru_cache

def construct():
      N = factorial(12)
      
      # Get all divisors of N
      divisors = []
      for i in range(1, int(N**0.5) + 1):
          if N % i == 0:
              divisors.append(i)
              if i != N // i:
                  divisors.append(N // i)
      divisors = sorted(divisors)
      
      # We want distinct factors > 12.
      # Filter divisors > 12
      valid_divs = [d for d in divisors if d > 12]
      
      # Binary search on the maximum factor M
      # Lower bound: 13
      # Upper bound: N
      lo, hi = 13, N
      best_M = N
      best_solution = None
      
      def dfs_check(rem, start_idx, M):
          # rem: remaining product to form
          # start_idx: index in valid_divs from which we can pick
          # M: current upper bound for factors
          
          if rem == 1:
              return True
              
          # Pruning:
          # If the smallest available divisor is > rem, we can't form rem.
          # Actually, we need rem to be divisible by some divisor.
          
          # Optimization: If rem is in valid_divs and rem >= valid_divs[start_idx] and rem <= M:
          # We can just take rem. But we must ensure we don't pick it if it's already used?
          # The start_idx mechanism ensures we only pick divisors with index > start_idx.
          # So if we pick rem, we are done.
          
          # Iterate through possible next factors
          for i in range(start_idx, len(valid_divs)):
              d = valid_divs[i]
              if d > M:
                  break
              if d > rem:
                  break
              if rem % d == 0:
                  # Check if we can form rem // d with factors > d (index > i) and <= M
                  if dfs_check(rem // d, i + 1, M):
                      return True
          return False
      
      # We need to retrieve the solution, not just check existence.
      # Modify dfs_check to return the list of factors.
      
      def find_solution(rem, start_idx, M, current_path):
          if rem == 1:
              return list(current_path)
              
          for i in range(start_idx, len(valid_divs)):
              d = valid_divs[i]
              if d > M:
                  break
              if d > rem:
                  break
              if rem % d == 0:
                  res = find_solution(rem // d, i + 1, M, current_path + [d])
                  if res is not None:
                      return res
          return None

      # Binary search
      while lo <= hi:
          mid = (lo + hi) // 2
          sol = find_solution(N, 0, mid, [])
          if sol is not None:
              best_M = mid
              best_solution = sol
              hi = mid - 1
          else:
              lo = mid + 1
              
      if best_solution is None:
          return [N]
      
      return best_solution
