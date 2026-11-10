import time

def construct():
       start_time = time.time()
       deadline = start_time + 50 # 50s limit
       answers = []
       
       for n in range(3, 257):
           if time.time() > deadline:
               # Fill remaining with None or best effort?
               # If we run out of time, we return what we have.
               # But we need 254 rows.
               # Append None for rest.
               while len(answers) < 254:
                   answers.append(None)
               return answers
           
           found = None
           # Try x starting from n//4 + 1
           # Limit x to avoid infinite loop if no solution (shouldn't happen)
           for x in range(n//4 + 1, n//4 + 100): 
               if time.time() > deadline:
                   break
               
               a = 4*x - n
               if a <= 0:
                   continue
               b = n*x
               
               # y range: (b/a, 2*b/a)
               # y_min = max(x+1, b//a + 1)
               # y_max = 2*b//a
               
               y_min = max(x + 1, b // a + 1)
               y_max = 2 * b // a
               
               # If range is too large, maybe skip? No, just check.
               # But if range is huge, it might be slow.
               # However, for small x, a is small, so range is large.
               # But we find solution quickly?
               
               for y in range(y_min, y_max + 1):
                   denom = a * y - b
                   if denom <= 0:
                       continue
                   if (b * y) % denom == 0:
                       z = (b * y) // denom
                       if z > y and z <= 10**12:
                           found = [x, y, z]
                           break
               
               if found:
                   break
           
           answers.append(found)
           
       return answers
