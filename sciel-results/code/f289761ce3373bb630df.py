         import heapq
         from fractions import Fraction
         import math

         def construct():
             target = Fraction(5, 31)
             rn, rd = 5, 31
             D_LIMIT = 10000000
             
             # Priority queue: (max_denom, rn, rd, path_tuple)
             # We use a counter to break ties if needed, but (max_denom, rn, rd) should be unique enough?
             # Actually, we want to process smaller max_denom first.
             
             heap = [(0, rn, rd, ())]
             # visited: (rn, rd) -> min_max_denom_seen
             visited = {}
             
             best_solution = None
             best_max = D_LIMIT + 1
             
             # Optional: Get a greedy solution to set an initial upper bound
             # Greedy:
             r = target
             greedy_path = []
             valid_greedy = True
             while r > 0 and len(greedy_path) < 10:
                 d = (r.denominator + r.numerator - 1) // r.numerator
                 if d > D_LIMIT:
                     valid_greedy = False
                     break
                 if d in greedy_path:
                     # Greedy might repeat? Unlikely for small fractions, but possible.
                     # If it repeats, this specific greedy path is invalid.
                     valid_greedy = False
                     break
                 greedy_path.append(d)
                 r -= Fraction(1, d)
             
             if valid_greedy and r == 0:
                 best_max = max(greedy_path)
                 best_solution = greedy_path
             else:
                 best_max = D_LIMIT + 1
                 best_solution = None

             while heap:
                 max_den, cur_rn, cur_rd, path = heapq.heappop(heap)
                 
                 # If we found a valid solution with this max, we can return immediately
                 # because heap is ordered by max_den.
                 if cur_rn == 0:
                     # Check distinctness
                     if len(set(path)) == len(path):
                         return list(path)
                     # If not distinct, continue searching for a distinct one
                     # But wait, if this path is not distinct, can a different path to the same state (0) be distinct?
                     # Yes. So we don't return. We just don't update best_solution.
                     # However, we must ensure we don't loop infinitely.
                     # The state (0,1) can be reached by many paths.
                     # We should mark (0,1) as visited?
                     # If we mark it, we might miss a distinct path with higher max?
                     # No, we want the MINIMUM max.
                     # If we pop (0,1) with max M, and it's invalid, we discard it.
                     # Any other path to (0,1) will have max >= M (since popped in order).
                     # So if the first valid distinct one is popped, it's optimal.
                     # We just need to keep popping until we find a distinct one.
                     continue
                     
                 # Pruning
                 if max_den >= best_max:
                     continue
                     
                 # Check visited
                 key = (cur_rn, cur_rd)
                 if key in visited:
                     if visited[key] <= max_den:
                         continue
                 else:
                     visited[key] = max_den
                     
                 # Generate next denominators
                 # Lower bound for d
                 # 1/d <= cur_rn/cur_rd  => d >= cur_rd/cur_rn
                 d_min = (cur_rd + cur_rn - 1) // cur_rn
                 
                 # Upper bound?
                 # If we pick d such that the new max_den becomes d, and d >= best_max, we can stop?
                 # The new max will be max(max_den, d).
                 # If d > best_max, the new state will be pruned immediately.
                 # So we only need to consider d < best_max.
                 d_max = best_max - 1
                 
                 # Also, we must ensure d is distinct from path?
                 # We check distinctness at the end.
                 # But to reduce branching, we can skip d if d in path.
                 # This is a strong pruning.
                 used_set = set(path)
                 
                 # Iterate d from d_min to d_max
                 # We should be careful with the range size.
                 # If d_max is large, this loop is slow.
                 # However, best_max starts from greedy (which might be large) or 10^7.
                 # If greedy was invalid, best_max is 10^7+1.
                 # Then d_max is 10^7.
                 # Iterating 10^7 values per node is too slow.
                 
                 # Optimization:
                 # We don't need to iterate ALL d.
                 # The new remainder is (cur_rn * d - cur_rd) / (cur_rd * d).
                 # Let num = cur_rn * d - cur_rd.
                 # Let den = cur_rd * d.
                 # We need num > 0 => d > cur_rd/cur_rn.
                 
                 # Is there a way to jump?
                 # For a fixed remainder, different d's lead to different remainders.
                 # We want to find a path to 0.
                 
                 # Heuristic: Try small d's first.
                 # But if best_max is 10^7, we can't scan all.
                 
                 # Let's refine the upper bound for d.
                 # If we pick d, the next remainder is smaller.
                 # If we pick d too large, the remainder becomes very small, forcing large next denominators.
                 # Specifically, if $r' = r - 1/d$, then $r' < 1/d - 1/(d+1) \approx 1/d^2$? No.
                 # $r = N/D$. $1/d \le N/D$.
                 $r' = (Nd - D) / (Dd)$.
                 If $d \approx D/N$, $Nd - D$ is small.
                 If $d$ is large, $Nd - D \approx Nd$. $r' \approx N/D$.
                 Wait, if $d$ is large, $1/d$ is small, so $r' \approx r$.
                 So picking large $d$ doesn't reduce the remainder much.
                 It just "wastes" a large denominator.
                 This suggests that optimal solutions use denominators close to the lower bound $D/N$.
                 
                 So we can limit the search for $d$ to a small range above $d_{min}$.
                 How small?
                 Empirically, for Egyptian fractions, the denominators are often close to the greedy choice.
                 Let's try $d$ from $d_{min}$ to $d_{min} + K$.
                 What is K?
                 If we miss the optimal path, we fail.
                 
                 Alternative:
                 Use the property that if $d$ is too large, the max denominator is likely to be larger than necessary.
                 Since we are doing Dijkstra, we pop states in order of max denom.
                 If we generate a state with max denom $M_{new}$, it goes into the heap.
                 If $M_{new}$ is large, it will be popped later.
                 We can just generate a reasonable number of candidates.
                 
                 Let's try generating $d$ in range $[d_{min}, d_{min} + 1000]$.
                 And also include $d_{min}$ specifically.
                 If the optimal solution requires a denominator $d$ much larger than $d_{min}$, this might fail.
                 But for $5/31$, the denominators are likely small.
                 
                 Let's check the 3-term solution again.
                 $a=7, b=13, c=31$? No.
                 $a=7, b=55, c=3979$? Max 3979.
                 Is there a solution with max < 3979?
                 $1/8 + 1/28 + 1/55 + \dots$?
                 
                 Let's just implement the Dijkstra with a limited range for d, say up to $d_{min} + 2000$ or until $d > best\_max$.
                 If we don't find a solution, we might need to expand the range or use a different strategy.
                 But given the "Minimize Largest Denominator" task, the solution is likely compact.

                 To be safe, we can also try a few specific "jump" denominators?
                 No, let's stick to the local range first.
                 
                 One more optimization:
                 If $cur\_rn == 1$, then we need $1/d = 1/cur\_rd \implies d = cur\_rd$.
                 So if $cur\_rn == 1$, we only need to check $d = cur\_rd$.
                 If $cur\_rd \le best\_max$ and $cur\_rd$ not in path, we can return.
                 
                 Similarly, if $cur\_rn$ is small, the number of divisors is limited?
                 No, $d$ doesn't have to divide anything directly in the next step, but the remainder simplifies.

                 Let's code this.
