      def construct():
          n = 64
          # GF(64) setup
          # mul function
          # log table
          colors = []
          for i in range(n):
              for j in range(i+1, n):
                  d = i ^ j
                  # d is non-zero
                  c = log64[d] % 3
                  colors.append(c)
          return {'n': n, 'colors': colors}
      