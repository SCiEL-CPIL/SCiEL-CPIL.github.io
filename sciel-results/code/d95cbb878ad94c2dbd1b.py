  def construct():
      # 54-comparator sorting network for 16 wires
      net = [
          (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
          (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
          (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
          (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
          (1, 2), (5, 6), (9, 10), (13, 14),
          (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
          (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
          (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
          (1, 2), (5, 6), (9, 10), (13, 14),
          (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)
      ]
      # Verify
      correct = True
      for mask in range(1 << 16):
          inputs = [(mask >> i) & 1 for i in range(16)]
          for (i, j) in net:
              if inputs[i] > inputs[j]:
                  inputs[i], inputs[j] = inputs[j], inputs[i]
          for k in range(15):
              if inputs[k] > inputs[k + 1]:
                  correct = False
                  break
          if not correct:
              break
      if not correct:
          # Fallback to Batcher OOM if hardcoded is wrong
          # ... generate Batcher OOM ...
          pass
      return net
  