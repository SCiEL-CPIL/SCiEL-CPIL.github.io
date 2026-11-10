  def _ils(bits, deadline, rng):
      best_bits = bits.copy()
      best_size = int(bits.sum())
      cur_bits = bits.copy()
      while time.time() < deadline:
          cur_bits = _greedy_add_polish(cur_bits, deadline)
          cur_size = int(cur_bits.sum())
          if cur_size > best_size:
              best_size = cur_size
              best_bits = cur_bits.copy()
          # kick
          ones = np.where(cur_bits[1:]==1)[0]+1
          if len(ones) < 2:
              break
          k = int(rng.integers(1, 4))  # 1-3
          if k >= len(ones):
              k = len(ones) - 1
          to_remove = rng.choice(ones, size=k, replace=False)
          for x in to_remove:
              cur_bits[x] = 0.0
      return best_bits, best_size
  