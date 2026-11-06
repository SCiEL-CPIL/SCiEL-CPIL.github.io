def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = []
    best_size = 0
    
    MAX_N = 4096
    MAX_SUM = 3 * MAX_N
    is_sum = bytearray(MAX_SUM + 1) # 0 or 1

    def can_add(x, S):
        # Check (x,x,x)
        if is_sum[3*x]: return False
        # Check (x,x,a)
        for a in S:
            if is_sum[2*x + a]: return False
        # Check (x,a,b)
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                if is_sum[x + a + b]: return False
        return True

    def add_to_sums(x, S):
        is_sum[3*x] = 1
        for a in S:
            is_sum[2*x + a] = 1
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                is_sum[x + a + b] = 1

    def build_set(order):
        S = []
        # Reset is_sum?
        # It's faster to create a new bytearray or clear it.
        # Clearing 12k bytes is fast.
        for i in range(MAX_SUM + 1):
            is_sum[i] = 0
            
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
                add_to_sums(x, S[:-1]) # S[:-1] is the set before adding x
        return S

    while time.time() < deadline:
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        current_set = build_set(candidates)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            if best_size >= 40:
                break

    return best_set
