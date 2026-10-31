   import time
   import random
   import sys

   # EVOLVE-BLOCK-START
   def construct(n_rows, k):
       start_time = time.time()
       deadline = start_time + 110
       
       n = k + 1
       
       def time_left():
           return deadline - time.time()
       
       def get_diffs(row):
           diffs = set()
           for i in range(len(row)):
               for j in range(i+1, len(row)):
                   diffs.add(row[j] - row[i])
           return diffs
       
       def is_valid_config(rows):
           used = set()
           for row in rows:
               diffs = get_diffs(row)
               if len(diffs) != len(row) * (len(row) - 1) // 2:
                   return False
               if used.intersection(diffs):
                   return False
               used.update(diffs)
           return True
       
       def objective(rows):
           # Minimize sum of entries, which correlates with min max
           return sum(sum(r) for r in rows)
       
       # 1. Initial Solution: Greedy
       def greedy_init():
           rows = []
           used_diffs = set()
           for _ in range(n_rows):
               row = [0]
               while len(row) < n:
                   # Try small increments
                   found = False
                   for x in range(row[-1] + 1, 200):
                       valid = True
                       local_diffs = set()
                       for a in row:
                           d = x - a
                           if d in used_diffs or d in local_diffs:
                               valid = False
                               break
                           local_diffs.add(d)
                       if valid:
                           row.append(x)
                           used_diffs.update(local_diffs)
                           found = True
                           break
                   if not found:
                       # Fallback: keep increasing until valid
                       x = row[-1] + 1
                       while True:
                           valid = True
                           local_diffs = set()
                           for a in row:
                               d = x - a
                               if d in used_diffs or d in local_diffs:
                                   valid = False
                                   break
                           if valid:
                               row.append(x)
                               used_diffs.update(local_diffs)
                               break
                           x += 1
               rows.append(row)
           return rows
       
       best_rows = greedy_init()
       best_score = objective(best_rows)
       
       # 2. Local Search
       # We will try to move elements to smaller positions.
       # To allow moves, we might need to temporarily increase a value to decrease others, 
       # but simple hill climbing on sum usually works for these small instances.
       
       current_rows = [r[:] for r in best_rows]
       current_score = best_score
       
       while time_left() > 5:
           # Pick random row and index
           r_idx = random.randint(0, n_rows - 1)
           i_idx = random.randint(1, n - 1) # Index 0 is always 0
           
           row = current_rows[r_idx]
           old_val = row[i_idx]
           
           # Determine valid range for new value
           min_val = row[i_idx - 1] + 1
           max_val = row[i_idx + 1] - 1 if i_idx < n - 1 else old_val + 10 # Allow slight expansion if needed? No, let's restrict to current bounds first.
           
           # If we only allow decreasing, we can get stuck. 
           # Let's try values in [min_val, max_val]
           # To be efficient, we only try to decrease the sum.
           # So we try values < old_val.
           
           # However, simply decreasing might break validity.
           # We need to check if the new value creates unique diffs.
           
           # Optimization: Precompute diffs for the current row excluding i_idx?
           # That's expensive. Let's just check on the fly.
           
           # Try a few random candidates below old_val
           candidates = []
           if old_val > min_val:
               # Sample some candidates
               for _ in range(10):
                   cand = random.randint(min_val, old_val - 1)
                   candidates.append(cand)
               candidates.append(min_val)
               candidates.append(old_val - 1)
           else:
               # If at minimum, try to increase? No, that increases sum.
               # Maybe skip this move.
               continue
           
           for cand in candidates:
               if cand == old_val:
                   continue
               
               # Check validity of new row
               new_row = row[:]
               new_row[i_idx] = cand
               
               # Check strictly increasing
               if not all(new_row[j] < new_row[j+1] for j in range(len(new_row)-1)):
                   continue
               
               # Check diffs
               new_diffs = get_diffs(new_row)
               if len(new_diffs) != n * (n - 1) // 2:
                   continue
               
               # Check global conflicts
               # We need to ensure new_diffs don't conflict with diffs from other rows
               # And don't conflict with "old" diffs from this row? 
               # Actually, we replace the row, so we just need to check against OTHER rows.
               
               conflict = False
               for other_idx, other_row in enumerate(current_rows):
                   if other_idx == r_idx:
                       continue
                   other_diffs = get_diffs(other_row)
                   if new_diffs.intersection(other_diffs):
                       conflict = True
                       break
               
               if not conflict:
                   # Accept move
                   current_rows[r_idx] = new_row
                   new_score = objective(current_rows)
                   
                   if new_score < best_score:
                       best_score = new_score
                       best_rows = [r[:] for r in current_rows]
                       
       return best_rows
   # EVOLVE-BLOCK-END
   