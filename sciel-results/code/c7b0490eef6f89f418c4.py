import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        # x >= floor(n/4) + 1
        x_min = n // 4 + 1
        # Also x must be at least 2 (since denominators are positive and distinct)
        # Try x values starting from x_min
        for x in range(x_min, min(x_min + 10000, LIMIT + 1)):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # y must satisfy: b/a < y <= 2b/a
            y_min = b // a + 1
            y_max = (2 * b) // a
            if y_max < y_min:
                continue
            # Also y > x for distinctness (since we assume x <= y <= z)
            if y_min <= x:
                y_min = x + 1
            if y_min > y_max:
                continue
            # Limit search to reasonable range to avoid infinite loops
            # For each y, z = b*y / (a*y - b)
            # We need a*y - b > 0 (guaranteed by y > b/a)
            # And z must be integer and z > y
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= LIMIT:
                        # Check distinctness: x, y, z all distinct
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        answers[idx] = found
    
    return answers
