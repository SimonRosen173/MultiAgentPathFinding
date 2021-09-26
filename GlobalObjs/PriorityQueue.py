import heapq


class PriorityQueue:
    def __init__(self):
        self.q = []

    def push(self, val, priority):
        heapq.heappush(self.q, (priority * -1, val))

    def pop(self):
        el = heapq.heappop(self.q)
        return el[0] * -1, el[1]

    def is_empty(self):
        return len(self.q) == 0
