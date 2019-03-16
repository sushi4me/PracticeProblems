class ExerciseProblems:

    # Implementation of merge sort
    @staticmethod
    def merge_sort(self, arr):
        # Variables
        result = []
        mid = len(arr)//2
        left = arr[mid:]
        right = arr[:mid]

        # Base case
        if len(arr) <= 1:
            return arr

        # Recursive call to merge_sort for left and right sides until base case
        left = ExerciseProblems.merge_sort(left)
        right = ExerciseProblems.merge_sort(right)

        # Compare and pop elements until one runs out
        while left and right:
            if left[0] < right[0]:
                result.append(right.pop(0))
            else:
                result.append(left.pop(0))

        # Pop the rest of the elements which are sorted
        while left:
            result.append(left.pop(0))
        while right:
            result.append(right.pop(0))

        return result

    # Finds a pair from the two lists yielding minimal difference
    def min_diff(self, list1, list2):
        # Variables
        min = None
        i = 0
        j = 0

        # The lists should be sorted
        list1 = self.merge_sort(list1)
        list2 = self.merge_sort(list2)

        # If the difference is 0, we are done
        while min != 0:
            # Check if the diff is less than the min found so far
            diff = abs(list1[i] - list2[j])
            if min is None or min > diff:
                min = diff

            # If we have went through both lists, we are done
            if i == len(list1)-1 and j == len(list2)-1:
                print("({}, {})".format(list1[i], list2[j]))
                return min

            # Shift the list which has the smaller value in attempt to close the diff gap
            if list1[i] > list2[j]:
                j += 1
            else:
                i += 1

        print("({}, {})".format(list1[i], list2[j]))
        return min    

    # Find the largest collection in a grid
    @staticmethod
    def find_largest_collection(grid):
        # Variables
        most_seen_thus_far = 1
        count = 0

        # Find out the number of columns and rows, verify it is a rectangular grid
        rows = len(grid)
        
        if len(set(map(len, grid))) == 1:
            columns = len(grid[0])
        else:
            raise ValueError("The grid dimensions are not completely rectangular!")

        # Create a grid to indicated visited or not
        visited = [[0 for i in range(columns)] for j in range(rows)]

        # Traverse the grid and mark visited spots
        nodes_to_visit = [(0, 0)]

        # While we have nodes to visit in the list
        while nodes_to_visit:
            count += 1
            i, j = nodes_to_visit.pop(0)
            print("Visiting: {}, {}".format(i, j))
            visited[i][j] = 1

            node_value = grid[i][j]

            # Pop adjacent grid nodes into the list to visit
            if i > 0 and grid[i-1][j] == node_value and not visited[i-1][j]:
                nodes_to_visit.append((i-1, j))
            if i < rows-1 and grid[i+1][j] == node_value and not visited[i+1][j]:
                nodes_to_visit.append((i+1, j))
            if j > 0 and grid[i][j-1] == node_value and not visited[i][j-1]:
                nodes_to_visit.append((i, j-1))
            if j < columns-1 and grid[i][j+1] == node_value and not visited[i][j+1]:
                nodes_to_visit.append((i, j+1))

            # Check if we have a new highest sequence
            most_seen_thus_far = count if count > most_seen_thus_far else most_seen_thus_far

            # Check if we hit a dead end for a specific node value, if so find the next sequence
            if not nodes_to_visit:
                # Reset count
                count = 0

                # Find a node that has yet to be visited
                for find_i, row in enumerate(visited):
                    try:
                        find_j = row.index(0)
                        break
                    except ValueError:
                        print("ValueError")
                        find_j = -1
                        continue

                # Append the next node to start from
                if find_j >= 0:
                    nodes_to_visit.append((find_i, find_j))
                else:
                    # find_j is not defined therefore we did not find an unvisited node
                    return most_seen_thus_far

        return most_seen_thus_far

    # Computes the Nth row of Pascal's triangle
    def pascal_row(self, row):
        pascal_row = [1]
        result_row = []

        # Calculate N rows of Pascal's
        for i in range(row):
            result_row = []

            # Compute N digits for Nth row
            for j in range(i + 1):
                # Compute the jth digit and append to row
                row_digit = self.compute_digit(pascal_row, j)
                result_row.append(row_digit)
            pascal_row = result_row

        return result_row

    # Helper function for pascal_row
    @staticmethod
    def compute_digit(pascal_row, index):
        """
        Take the indexed digit and the previous digit to get the resultant for the next row. The corner cases can be found
        if the index is zero or greater than the length of the previous row.
        """
        if index - 1 >= 0 and index <= len(pascal_row) - 1:
            return pascal_row[index] + pascal_row[index - 1]
        else:
            return 1

    # Finds the longest palindrome able to be made given a sequence of characters
    @staticmethod
    def longest_palindrome(string_input):
        string_map = {}
        max_palindrome = 0

        for i in range(len(string_input)):
            if string_input[i] not in string_map:
                string_map[string_input[i]] = 1
            else:
                string_map[string_input[i]] += 1

        for char in string_map:
            max_palindrome += string_map[char]//2

        if len(string_input)/2 > max_palindrome:
            max_palindrome += 1

        return max_palindrome

    # Find out if a pair in the array adds up to sum K
    @staticmethod
    def k_sum(arr, k):
        compliment_map = set()

        for i in range(len(arr)):
            compliment = k - arr[i]

            if arr[i] in compliment_map:
                return True
            else:
                compliment_map.add(compliment)

        return False

    # TODO
    @staticmethod
    def master_mind(guess, solution):
        hits = 0
        pseudo_hits = 0
        pseudo_list = {}

        if len(guess) != len(solution):
            return 0, 0

        # Check for hits -- matching indices between guess and solution
        for i in range(len(guess)):
            if guess[i] == solution[i]:
                hits += 1
            else:
                if guess[i] in pseudo_list:
                    pseudo_list[guess[i]] += 1
                else:
                    pseudo_list[guess[i]] = 1

        # Check for pseudo hits
        for i in range(len(guess)):
            if guess[i] != solution[i] and solution[i] in pseudo_list:
                pseudo_list[solution[i]] -= 1
                pseudo_hits += 1

        return hits, pseudo_hits

    # Finds the smallest sub-array which can be sorted to produce a fully sorted array (ascending)
    @staticmethod
    def sub_sort(arr):
        left_end = -1
        right_start = -1

        # Find the tentative end of the left-hand side
        for i in range(1, len(arr)):
            if arr[i] >= arr[i - 1]:
                continue
            else:
                left_end = i - 1
                print("The largest left is: {}".format(arr[i-1]))
                break

        # Find the tentative start of the right-hand side
        for i in range(len(arr) - 2, 0, -1):
            if arr[i] <= arr[i + 1]:
                continue
            else:
                right_start = i + 1
                print("The smallest right is: {}".format(arr[i+1]))
                break

        # Search for the min and max of the middle array
        middle_arr = arr[left_end + 1: right_start]

        mid_min = right_start
        mid_max = left_end

        for i in range(left_end + 1, right_start):
            # The right_start should be GREATER than all other elements in the middle and left
            if arr[i] < arr[mid_min]:
                mid_min = i
            # The left_end should be LESS than all other elements in the middle and right
            if arr[i] > arr[mid_max]:
                mid_max = i

        print("Middle array min/max: {}, {}".format(arr[mid_min], arr[mid_max]))

        # Adjust left_end to be greater than the largest found from middle and right
        while left_end > 0 and arr[left_end] > arr[mid_min]:
            left_end -= 1

        # Adjust right_start to be less than the smallest found from middle and left
        while right_start < len(arr) - 1 and arr[right_start] < arr[mid_max]:
            right_start += 1

        return left_end + 1, right_start - 1


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def add(self, left=None, right=None):
        if left:
            print("Added: {}".format(left.val))
            self.left = left
        if right:
            print("Added: {}".format(right.val))
            self.right = right


# Converts a binary search tree into a max heap
def k_sum_binary_tree(root):

    def inorder_traversal(root, arr):
        if root is None:
            return None

        inorder_traversal(root.left, arr)
        arr.append(root.value)
        inorder_traversal(root.right, arr)

        return arr

    def pop_max_heap(arr):
        max_value = arr[0]
        arr[:] = reheapify_max_heap(arr)
        return max_value

    def reheapify_max_heap(arr):
        arr[0] = arr[-1]
        arr.pop(-1)
        index = 1
        swap = 0

        while index * 2 <= len(arr) or index * 2 + 1 <= len(arr):
            if index * 2 > len(arr):
                swap = index * 2 + 1
            elif index * 2 + 1 > len(arr):
                swap = index * 2
            elif arr[index * 2 - 1] < arr[index * 2]:
                swap = index * 2 + 1
            elif arr[index * 2 - 1] >= arr[index * 2]:
                swap = index * 2

            print("{} < {}?".format(arr[index-1], arr[swap-1]))

            if arr[index - 1] < arr[swap - 1]:
                temp = arr[index - 1]
                arr[index - 1] = arr[swap - 1]
                arr[swap - 1] = temp
                index = swap
            else:
                break

        return arr

    arr = []
    arr = inorder_traversal(root, arr)
    arr.reverse()
    print(arr)
    print(pop_max_heap(arr))
    print(arr)


def test_node():
    node = Node('root', Node('left', Node('left.left')), Node('right'))
    assert deserialize(serialize(node)).left.left.val == 'left.left'


def input_value(arr, index, value):
    """
    Insert the value at the specified index but insert -1 for all other indices passed that are empty.
    """
    i = 0
    while i < index + 1:
        try:
            if i == index:
                arr[index] = value
            i += 1
        except IndexError:
            arr.insert(i, -1)
            i -= 1

    return arr


def serialize(node):
    """
    Function converts a binary tree into a string (indices correlates to the binary tree)
    """
    tree_list = []
    traverse_list = []
    index_list = []
    index = 1
    current_node = node

    while current_node:
        tree_list = input_value(tree_list, index, current_node.val)

        if current_node.right:
            tree_list = input_value(tree_list, index * 2 + 1, current_node.right.val)
            traverse_list.append(current_node.right)
            index_list.append(index * 2 + 1)

        if current_node.left:
            tree_list = input_value(tree_list, index * 2, current_node.left.val)
            traverse_list.append(current_node.left)
            index_list.append(index * 2)

        if traverse_list:
            current_node = traverse_list.pop()
            index = index_list.pop()
        else:
            current_node = None

    serialized_string = ','.join(str(element) for element in tree_list)

    return serialized_string


def deserialize(serialized_string):
    """
    Function converts a serialized string into a binary tree (indicies correlates to the binary tree)
    """
    tree_list = serialized_string.split(',')

    def _recursive(index):
        if tree_list[index] == -1:
            return None

        value = tree_list[index]
        left, right = None, None

        if index * 2 < len(tree_list):
            left = _recursive(index * 2)

        if index * 2 + 1 < len(tree_list):
            right = _recursive(index * 2 + 1)

        return Node(value, left, right)

    def _nonrecursive(index):
        root = Node(tree_list[index])
        visit = [root]

        for i in range(1, len(tree_list)):
            current_node = visit.pop()
            print(current_node)

            if i * 2 + 1 < len(tree_list) and tree_list[i * 2 + 1] != -1:
                current_node.right = Node(tree_list[i * 2 + 1])
                visit.append(current_node.right)

            if i * 2 < len(tree_list) and tree_list[i * 2] != -1:
                current_node.left = Node(tree_list[i * 2])
                visit.append(current_node.left)

        return root

    return _nonrecursive(1)

