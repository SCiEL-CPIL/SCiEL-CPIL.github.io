def construct():
          result = []
          for i in range(3):
              for j in range(3):
                  result.append([i, 3+j])
          for i in range(3):
              for j in range(3):
                  result.append([6+i, 9+j])
          return result
