      def count_violations_fast(masks, n):
          # masks is a list of ints
          # n is the number of vertices
          # We assume vertices are 0..n-1
          # in_masks[i] has bit j set if j -> i
          # Active mask is (1 << n) - 1
          active_mask = (1 << n) - 1
          violations = 0
          for i in range(n):
              m1 = masks[i]
              for j in range(i+1, n):
                  m2 = masks[j]
                  inter12 = m1 & m2 & active_mask
                  if inter12 == 0: continue
                  for k in range(j+1, n):
                      m3 = masks[k]
                      inter123 = inter12 & m3
                      if inter123 == 0: continue
                      for l in range(k+1, n):
                          m4 = masks[l]
                          if (inter123 & m4) == 0:
                              violations += 1
          return violations
      