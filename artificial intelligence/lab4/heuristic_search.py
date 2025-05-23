import heapq
import time
import math

# def heuristic(state): # 曼哈顿距离
#     h = 0
#     for i in range(4):
#         for j in range(4):
#             num = state[i][j]
#             if num != 0:
#                 target_row = (num - 1) // 4
#                 target_col = (num - 1) % 4
#                 h += abs(i - target_row) + abs(j - target_col)
#     return h


# def heuristic(state): #不在目标位置的方格数
#     h = 0
#     for i in range(4):
#         for j in range(4):
#             num = state[i][j]
#             if num != 0:
#                 target_row = (num - 1) // 4
#                 target_col = (num - 1) % 4
#                 if i != target_row or j != target_col:
#                     h += 1
#     return h

def heuristic(state): # 欧几里得距离
    h = 0
    for i in range(4):
        for j in range(4):
            num = state[i][j]
            if num != 0:
                target_row = (num - 1) // 4
                target_col = (num - 1) % 4
                h += math.sqrt((i - target_row)**2 + (j - target_col)**2)
    return h

class Node:
    def __init__(self, state, zero_pos, g, h, parent=None, moved_num=None):
        self.state = state
        self.zero_pos = zero_pos
        self.g = g
        self.h = h
        self.parent = parent
        self.moved_num = moved_num

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

def issolvable(state):
    flat = [num for row in state for num in row if num != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j] and flat[j] != 0 and flat[i] != 0 and flat[j] != 0:
                inversions += 1 
    for i in range(4):
        if 0 in state[i]:
            zero_row = i + 1
            break
    return (inversions + zero_row) % 2 == 0

def A_star(initial_state):
    start_time = time.time()
    initial_state_tuple = tuple(tuple(row) for row in initial_state)
    if not issolvable(initial_state_tuple):
        return None

    target_state = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 0)
    )
    if initial_state_tuple == target_state:
        return []

    zero_pos = None
    for i in range(4):
        for j in range(4):
            if initial_state_tuple[i][j] == 0:
                zero_pos = (i, j)
                break
        if zero_pos:
            break

    h_initial = heuristic(initial_state_tuple)
    initial_node = Node(initial_state_tuple, zero_pos, 0, h_initial)
    heap = []
    heapq.heappush(heap, initial_node)
    open = {initial_state_tuple: initial_node}
    close = {}
    count = 1

    while heap:
        print(count)
        count+=1
        current_node = heapq.heappop(heap)
        if current_node.state not in open:
            continue

        del open[current_node.state]

        if current_node.state == target_state:
            solution = []
            while current_node.parent is not None:
                solution.append(current_node.moved_num)
                current_node = current_node.parent
            solution.reverse()
            end_time = time.time()
            print(f"A*算法运行时间: {end_time - start_time:.8f}秒")
            return solution
        
        close[current_node.state] = current_node
        i, j = current_node.zero_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < 4 and 0 <= new_j < 4:
                num = current_node.state[new_i][new_j]
                state_list = [list(row) for row in current_node.state]
                state_list[i][j], state_list[new_i][new_j] = state_list[new_i][new_j], state_list[i][j]
                new_state = tuple(tuple(row) for row in state_list)
                # target_row = (num - 1) // 4
                # target_col = (num - 1) % 4
                # dx_old = abs(new_i - target_row) + abs(new_j - target_col)
                # dx_new = abs(i - target_row) + abs(j - target_col)
                # h_child = current_node.h - dx_old + dx_new
                g_child = current_node.g + 1
                child_node = Node(new_state, (new_i, new_j), g_child, heuristic(new_state), current_node, num)

                if new_state not in close and new_state not in open:
                    heapq.heappush(heap, child_node)
                    open[new_state] = child_node

                elif new_state in open:
                    existing_node = open[new_state]
                    if g_child < existing_node.g:
                        # existing_node.g = g_child
                        # existing_node.h = h_child
                        # existing_node.parent = current_node
                        # existing_node.moved_num = num
                        # existing_node.zero_pos = (new_i, new_j)
                        # heapq.heapify(heap)
                        heapq.heappush(heap, child_node)
                        open[new_state] = child_node  
                elif new_state in close:
                    existing_node = close[new_state]
                    if g_child < existing_node.g:
                        heapq.heappush(heap, child_node)
                        open[new_state] = child_node
                        del close[new_state]
    end_time = time.time()
    print(f"A*算法运行时间: {end_time - start_time:.8f}秒")
    return None


def IDA_star(initial_state):
    start_time = time.time()

    initial_state_tuple = tuple(tuple(row) for row in initial_state)
    if not issolvable(initial_state_tuple):
        return None

    target_state = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 0)
    )
    if initial_state_tuple == target_state:
        return []

    # Find initial zero position
    zero_pos = None
    for i in range(4):
        for j in range(4):
            if initial_state_tuple[i][j] == 0:
                zero_pos = (i, j)
                break
        if zero_pos:
            break

    # Initialize
    h_initial = heuristic(initial_state_tuple)
    c = 100  # Initial cost threshold
    initial_node = Node(initial_state_tuple, zero_pos, 0, h_initial)
    count = 1
    while True:
        stack = [initial_node]
        c_prime = float('inf')

        while stack:
            print(count)
            count += 1
            current_node = stack.pop()
            
            if current_node.state == target_state:
                # Reconstruct solution path
                solution = []
                while current_node.parent is not None:
                    solution.append(current_node.moved_num)
                    current_node = current_node.parent
                solution.reverse()
                end_time = time.time()
                print(f"IDA*算法运行时间: {end_time - start_time:.8f}秒")
                return solution
            
            # Generate child nodes
            i, j = current_node.zero_pos
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for di, dj in directions:
                new_i, new_j = i + di, j + dj
                if 0 <= new_i < 4 and 0 <= new_j < 4:
                    # Create new state
                    new_state = [list(row) for row in current_node.state]
                    new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
                    new_state_tuple = tuple(tuple(row) for row in new_state)
                    
                    # Skip parent state to avoid going back
                    if current_node.parent and new_state_tuple == current_node.parent.state:
                        continue
    
                    moved_num = current_node.state[new_i][new_j]
                    target_row = (moved_num - 1) // 4
                    target_col = (moved_num - 1) % 4
                    dx_old = abs(new_i - target_row) + abs(new_j - target_col)
                    dx_new = abs(i - target_row) + abs(j - target_col)
                    h_child = current_node.h - dx_old + dx_new
                    
                    child_node = Node(new_state_tuple, (new_i, new_j), current_node.g + 1, h_child, current_node, moved_num)
                    
                    if child_node.f() <= c:
                        stack.append(child_node)
                    else:
                        c_prime = min(c_prime, child_node.f())
    
        if not stack and c_prime == float('inf'):
            end_time = time.time()
            print(f"IDA*算法运行时间: {end_time - start_time:.8f}秒")
            return None  # No solution exists
        elif not stack and c_prime < float('inf'):
            c = c_prime  # Update the cost threshold for next iteration

test1 = [[1,2,4,8],[5,7,11,10],[13,15,0,3],[14,6,9,12]] ## 0.0096  0.0038 0.0030           0.4173      0.0563
test2 = [[2,5,1,3],[7,11,6,4],[10,14,9,8],[13,0,12,15]] ## 1.4363 2.1720 2.0900            144.5075    14.0416
test3 = [[5,3,7,8],[1,2,11,4],[13,6,15,14],[0,10,9,12]] ## 58.5407 177.0541 169.7496
test4 = [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]] ## 50.4318 93.7775
test5 = [[5,1,3,4],[2,7,8,12],[9,6,11,15],[0,13,10,14]] ## 0.0007 0.0001                    0.0226     0.0086
test6 = [[6,10,3,15],[14,8,7,11],[5,1,0,2],[13,12,9,4]] ## 100 seconds 397
test7 = [[11,3,1,7],[4,6,8,2],[15,9,10,13],[14,12,5,0]]
test8 = [[0,5,15,14],[7,9,6,13],[1,2,12,10],[8,11,4,3]]
print(A_star(test5))