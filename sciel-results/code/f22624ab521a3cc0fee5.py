import heapq

def construct():
    """
    Returns the first 20 5-smooth numbers (Hamming numbers).
    These are numbers of the form 2^i * 3^j * 5^k.
    They are chosen to have a small product set (since products are also 5-smooth)
    and a relatively small sumset (since they are dense small integers).
    """
    # Use a min-heap to generate the first 20 5-smooth numbers
    # Start with 1
    # To avoid duplicates, we can use a set or standard generation algorithm
    
    smooths = set()
    heap = [1]
    smooths_list = []
    
    while len(smooths_list) < 20:
        x = heapq.heappop(heap)
        if x in smooths:
            continue
        smooths.add(x)
        smooths_list.append(x)
        
        heapq.heappush(heap, x * 2)
        heapq.heappush(heap, x * 3)
        heapq.heappush(heap, x * 5)
        
    return smooths_list
