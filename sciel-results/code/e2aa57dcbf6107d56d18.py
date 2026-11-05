from fractions import Fraction
import time

def construct():
      start_time = time.time()
      deadline = start_time + 50
      
      target = Fraction(5, 31)
      best_max = float('inf')
      best_sol = None
      
      # 3-term search
      for a in range(7, 100): # a must be small
          if time.time() > deadline: break
          R_a = target - Fraction(1, a)
          if R_a <= 0: continue
          b_min = int(1 / R_a) + 1
          b_max = int(2 / R_a)
          for b in range(b_min, b_max + 1):
              if time.time() > deadline: break
              if b <= a: continue
              R_ab = R_a - Fraction(1, b)
              if R_ab <= 0: continue
              c = 1 / R_ab
              if c.denominator == 1:
                  c_int = c.numerator
                  if c_int > b:
                      if c_int < best_max:
                          best_max = c_int
                          best_sol = [a, b, c_int]
      
      # 4-term search
      # a < b < c < d
      # 1/a + 1/b + 1/c + 1/d = 5/31
      # c range for given a, b:
      # 1/c < R_ab and 2/c > R_ab => c in (1/R_ab, 2/R_ab)
      
      for a in range(7, 100):
          if time.time() > deadline: break
          R_a = target - Fraction(1, a)
          if R_a <= 0: continue
          # b must satisfy 1/b < R_a and 2/b > R_a ? No, b is second term.
          # 1/b + 1/c + 1/d = R_a.
          # Max sum for b, c, d is 1/b + 1/(b+1) + 1/(b+2).
          # Min sum is ...
          # We just iterate b.
          # 1/b < R_a => b > 1/R_a.
          b_min = int(1 / R_a) + 1
          # Upper bound for b?
          # If b is too large, 1/b is small.
          # We need 1/b + 1/c + 1/d = R_a.
          # Since c > b, d > c, 1/c < 1/b, 1/d < 1/c.
          # 3/b > 1/b + 1/c + 1/d = R_a => b < 3/R_a.
          b_max = int(3 / R_a)
          
          for b in range(b_min, b_max + 1):
              if time.time() > deadline: break
              if b <= a: continue
              R_ab = R_a - Fraction(1, b)
              if R_ab <= 0: continue
              
              # Now 1/c + 1/d = R_ab.
              # c > b.
              # 1/c < R_ab => c > 1/R_ab.
              # 2/c > R_ab => c < 2/R_ab.
              c_min = int(1 / R_ab) + 1
              c_max = int(2 / R_ab)
              
              # Also c must be > b.
              if c_min <= b: c_min = b + 1
              
              for c in range(c_min, c_max + 1):
                  if time.time() > deadline: break
                  R_abc = R_ab - Fraction(1, c)
                  if R_abc <= 0: continue
                  d = 1 / R_abc
                  if d.denominator == 1:
                      d_int = d.numerator
                      if d_int > c:
                          if d_int < best_max:
                              best_max = d_int
                              best_sol = [a, b, c, d_int]
      
      if best_sol is None:
          return [8, 31, 248]
      return best_sol
