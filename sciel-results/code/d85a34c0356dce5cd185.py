      import itertools

      def construct(n):
          if n <= 1:
              return "1"
          if n == 2:
              return "121"
          
          perms = list(itertools.permutations(range(1, n+1)))
          perm_to_idx = {p: i for i, p in enumerate(perms)}
          num_perms = len(perms)
          full_mask = (1 << num_perms) - 1
          
          # State: (last_n_tuple, mask, parent_id, symbol)
          # parent_id = -1 for start
          
          states = []
          # Start state
          states.append(((), 0, -1, 0)) # symbol 0 is dummy
          beam = [0] # indices into states
          
          for step in range(1, n * num_perms + 10):
              next_beam = []
              for state_id in beam:
                  last_n, mask, parent, sym = states[state_id]
                  for c in range(1, n+1):
                      # Update last_n
                      if len(last_n) < n:
                          new_last_n = last_n + (c,)
                      else:
                          new_last_n = last_n[1:] + (c,)
                          
                      new_mask = mask
                      if len(new_last_n) == n:
                          if new_last_n in perm_to_idx:
                              idx = perm_to_idx[new_last_n]
                              if not (mask >> idx) & 1:
                                  new_mask = mask | (1 << idx)
                                  
                      if new_mask == full_mask:
                          # Found solution
                          # Reconstruct
                          path = []
                          curr = len(states)
                          states.append((new_last_n, new_mask, state_id, c))
                          curr = len(states) - 1
                          while curr != -1:
                              _, _, p, s = states[curr]
                              if s != 0:
                                  path.append(s)
                              curr = p
                          return "".join(map(str, reversed(path)))
                          
                      states.append((new_last_n, new_mask, state_id, c))
                      next_beam.append(len(states) - 1)
              
              if not next_beam:
                  break
                  
              # Prune
              # Sort by mask popcount
              # We need to access states[next_beam[i]]
              def get_score(idx):
                  return states[idx][1].bit_count()
              
              next_beam.sort(key=get_score, reverse=True)
              beam = next_beam[:100]
              
          # Fallback
          # Return best in beam
          best_id = beam[0]
          path = []
          curr = best_id
          while curr != -1:
              _, _, p, s = states[curr]
              if s != 0:
                  path.append(s)
              curr = p
          return "".join(map(str, reversed(path)))
      