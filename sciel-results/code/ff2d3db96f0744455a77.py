      for j in range(4): # j is local index 0..3
          for c in range(7):
              # Block 0
              V[j][c] = V_block[j][c]
              # Block 1
              V[j + 4][c + 7] = V_block[j][c]
      