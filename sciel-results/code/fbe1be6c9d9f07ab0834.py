<<<<<<< SEARCH
            if mode == "single":
                # Replace one element
                i = random.randrange(20)
                x = A[i]
                cands = set()
                # Multiplicative neighbors (exploit lattice structure)
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Additive neighbors
                for d in [-3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
                    nv = x + d
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Random
                cands.add(random.randrange(1, 10001))
                # Lattice-inspired: multiply/divide by 2 or 3
                for div_mult in [2, 3, 5]:
                    cands.add(x * div_mult)
                    if x % div_mult == 0:
                        cands.add(x // div_mult)
                cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                if not cands:
                    continue
                nv = random.choice(cands)
                old = A[i]
                A[i] = nv
                new_obj = objective(A)
                if new_obj < cur_obj:
                    cur_obj = new_obj
                    stuck_count = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                elif new_obj == cur_obj and random.random() < 0.1:
                    # Allow neutral moves occasionally to escape plateaus
                    stuck_count = 0
                else:
                    A[i] = old
                    stuck_count += 1
=======
            if mode == "single":
                # Replace one element
                i = random.randrange(20)
                x = A[i]
                cands = set()
                # Multiplicative neighbors (exploit lattice structure)
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Additive neighbors
                for d in [-3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
                    nv = x + d
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Random
                cands.add(random.randrange(1, 10001))
                # Lattice-inspired: multiply/divide by 2, 3, or 5
                for div_mult in [2, 3, 5]:
                    cands.add(x * div_mult)
                    if x % div_mult == 0:
                        cands.add(x // div_mult)
                # Also try nearby lattice points
                # Find closest lattice points to x
                for i_exp in range(0, 14):
                    for j_exp in range(0, 10):
                        v = (2 ** i_exp) * (3 ** j_exp)
                        if v <= 10000 and abs(v - x) <= 100:
                            cands.add(v)
                cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                if not cands:
                    continue
                nv = random.choice(cands)
                old = A[i]
                A[i] = nv
                new_obj = objective(A)
                if new_obj < cur_obj:
                    cur_obj = new_obj
                    stuck_count = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                elif new_obj == cur_obj and random.random() < 0.15:
                    # Allow neutral moves occasionally to escape plateaus
                    stuck_count = 0
                else:
                    A[i] = old
                    stuck_count += 1
>>>>>>> REPLACE

