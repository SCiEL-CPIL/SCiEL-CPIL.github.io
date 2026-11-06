  candidates = [v for v in fast_mem if v not in outputs]
  if not candidates:
      candidates = list(fast_mem)
  