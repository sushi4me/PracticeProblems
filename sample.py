########################################################################################################################

class ExerciseProblems():

    # Implementation of merge sort
    @staticmethod
    def merge_sort(arr):
        # Variables
        result = []
        mid = len(arr)//2
        left = arr[mid:]
        right = arr[:mid]

        # Base case
        if len(arr) <= 1:
            return arr

        # Resursive call to merge_sort for left and right sides until base case
        left = merge_sort(left)
        right = merge_sort(right)

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
    @staticmethod
    def min_diff(list1, list2):
        # Variables
        min = None
        i = 0
        j = 0

        # The lists should be sorted
        list1 = merge_sort(list1)
        list2 = merge_sort(list2)

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

    # Finds the largest connected collection in a grid
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

            # Check if we hit a deadend for a specific node value, if so find the next sequence
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
    @staticmethod
    def pascal_row(row):
        pascal_row = [1]
        result_row = []

        # Calculate N rows of Pascal's
        for i in range(row):
            result_row = []

            # Compute N digits for Nth row
            for j in range(i + 1):
                # Compute the jth digit and append to row
                row_digit = compute_digit(pascal_row, j)
                result_row.append(row_digit)
            pascal_row = result_row

        return result_row

    @staticmethod
    def compute_digit(pascal_row, index):
        '''
        Take the indexed digit and the previous digit to get the resultant for the next row. The corner cases can be found
        if the index is zero or greater than the length of the previous row.
        '''
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
            return (0, 0)

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

        return (hits, pseudo_hits)

    # Finds the smallest sub-array which can be sorted to produce a fully sorted array (ascending)
    @staticmethod
    def sub_sort(arr):
        left_end = -1
        right_start = -1
        mid_min = None
        mid_max = None

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

        return (left_end + 1, right_start - 1)

########################################################################################################################

class DailyCodingProblem():

    ## From Daily Coding Problem

    # Problem 4: First missing positive integer [Hard]
    @staticmethod
    def smallest_positive_int(arr):
        for i, element in enumerate(arr):
            # While the element does not equal its indexing and is (0, len(arr)].
            # while i + 1 != element and 0 < arr[i] <= len(arr):
                # Swap the values
                arr[i], arr[element - 1] = arr[element - 1], arr[i]

                # Check if we are swapping identical values so we are not caught in a loop
                if arr[i] == arr[element - 1]:
                    break

        # Go through the array and find the first value which does not match its indexing
        for i, element in enumerate(arr):
            if i + 1 != element:
                return i + 1

        return len(arr) + 1

    # Problem 7: Number of ways decoded [Medium]
    @staticmethod
    def decode_numerical_string(encoded_string):
        possible_double = 0
        possible_double_offset = 0

        # Start by finding each of the possible decoding scheme by pairing, starting 0 index
        for i in range(0, len(encoded_string) - 1, 2):
            if 9 < int(encoded_string[i] + encoded_string[i + 1]) < 27:
                possible_double += 1

        # This starts on an offset since each digit can be paired with the prior or next digit, starting 1 index
        for i in range(1, len(encoded_string) - 1, 2):
            if 9 < int(encoded_string[i] + encoded_string[i + 1]) < 27:
                possible_double_offset += 1

        # For each valid double-digit code there is a case in which it is "on" or "off", therefore the possible outcomes
        # can be found using a power of 2.  We subtract 1 in order to account for overcounting in the case they are all
        # "off" -- entire string is treated as single digit codes.
        return (2)**possible_double + (2)**possible_double_offset - 1

    # Problem 8: Unival tree [Easy]
    @staticmethod
    def is_unival_test_case(tree_node):
        t = TreeNode(0, TreeNode(1, TreeNode(1), TreeNode(1)), TreeNode(0, TreeNode(0), TreeNode(0)))

        node_list = t.convert_tree()

        return t.check_unival(node_list)

    # Problem 9: Largest sum of non-adjacent [Hard]
    @staticmethod
    def contiguous_sequence(arr):
        sum_so_far = 0
        max_sum = 0
        sub_array = []

        for i in range(len(arr)):
            sum_so_far += arr[i]
            sub_array.append(arr[i])

            # Check if we have a new highest sum
            if sum_so_far > max_sum:
                max_sum = sum_so_far

            # Reset the sum thus far if the sum drops below zero (e.g. [2, -3])
            if sum_so_far < 0:
                sum_so_far = 0
                sub_array = []

        return sub_array

    # Problem 11: Autocomplete [Hard]
    @staticmethod
    def autocomplete_test_case(s):
        words = ['deer', 'deal', 'dealt', 'dear', 'dog']
        t = Trie()

        for word in words:
            t.insert(word)

        suffixes = t.elements('de')

        return [prefix + w for w in suffixes]

    # Problem 12: Possible steps [Hard]
    @staticmethod
    def possible_steps(total_steps, steps_set):
        # The index corresponds to the step
        cache = [0 for _ in range(total_steps + 1)]

        # There is one way to get to the zero step
        cache[0] = 1

        # For each step of the way we find the number of ways we can reach the previous step assuming we took S to 
        # arrive
        for i in len(cache):
            cache[i] += sum(cache[i - s] for s in steps_set if i - s >= 0)

        return cache[-1]

    # Problem 13: k-Distinct characters longest substring [Hard]
    @staticmethod
    def k_distinct_substring(s, k):
        start_index = 0
        end_index = 0
        max_length = 0
        count = [0] * 26

        def is_valid(count, k):
            unique = 0

            for e in count:
                if e > 0:
                    unique += 1

            return unique <= k

        #count[ord(s[0]) - ord('a')] += 1

        for i in range(0, len(s)):
            count[ord(s[i]) - ord('a')] += 1
            end_index += 1

            # Verify our current window
            while not is_valid(count, k):
                count[ord(s[start_index]) - ord('a')] -= 1
                start_index += 1

            if end_index - start_index + 1 - 1 > max_length:
                max_length = end_index - start_index + 1 - 1

            #print("{}, {}".format(start_index, end_index))

        return max_length

    # Problem 16: Implementing Log API [Easy]
    @staticmethod
    def log_api():
        pass

    # Problem 17: Parse Directory [Hard]
    @staticmethod
    def parse_directory():
        pass

# Class for Problem 11: Autocomplete [Hard]
class Trie(object):

    def __init__(self):
        self.values = {}

    def insert(self, prefix):
        temp = self.values

        for char in prefix:
            if char not in temp:
                temp[char] = {}
            temp  = temp[char]

        temp['__END'] = True

    def elements(self, prefix):
        temp = self.values

        for char in prefix:
            if char in temp:
                temp = temp[char]
            else:
                return []

        return self._elements(temp)

    def _elements(self, values):
        result = []

        # For each of the keys in the trie concatenate the children to 
        for key, value in values.items():
            if key == '__END':
                subresult = ['']
            else:
                subresult = [key + s for s in self._elements(value)]
            result.extend(subresult)

        return result

# Class for Problem 8: Number of unival trees [Easy]
class TreeNode(object):

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.visited = False
        
    @staticmethod
    def count_unival_recursive(node):
        # The node does not exist, it is unival but does not contribute to count
        if node is None:
            return 0, True

        left_count, left_unival = count_unival_recursive(node.left)

        right_count, right_unival = count_unival_recursive(node.right)

        total_count = left_count + right_count

        if left_unival and right_unival:
            if node.left is not None and node.value != node.left.value:
                return total_count, False
            if node.right is not None and node.value != node.right.value:
                return total_count, False
            return total_count + 1, True
        return total_count, False

    def convert_tree(self):
        '''
        Converts a tree into a dictionary with indexes as their keys
        '''
        node_list = {}
        index = 1
        visit_list = [self]
        current_node = None

        while visit_list:
            current_node = visit_list.pop(0)
            if current_node is None:
                index += 1
                continue

            node_list[index] = {'value': current_node.value, 'unival': None}

            if current_node.left:
                visit_list.append(current_node.left)
            else:
                visit_list.append(None)
            if current_node.right:
                visit_list.append(current_node.right)
            else:
                visit_list.append(None)

            index += 1

        return node_list

    @staticmethod
    def check_unival(node_list):
        total_unival = 0
        current_node = None

        for key in sorted(node_list.keys(), reverse=True):
            if key * 2 not in node_list and key * 2 + 1 not in node_list:
                total_unival += 1
                node_list[key]['unival'] = True
                continue

            if key * 2 in node_list:
                unival_left = node_list[key * 2]['unival'] if node_list[key * 2]['value'] == node_list[key]['value'] else False
            else:
                unival_left = True

            if key * 2 + 1 in node_list:
                unival_right = node_list[key * 2 + 1]['unival'] if node_list[key * 2 + 1]['value'] == node_list[key]['value'] else False
            else:
                unival_right = True

            if unival_left and unival_right:
                node_list[key]['unival'] = True
                total_unival += 1
            else:
                node_list[key]['unival'] = False

        return total_unival

########################################################################################################################

class MaxHeapNode():

    def __init__(self, value, index):
        self.parent = None
        self.left = None
        self.right = None
        self.value = value
        self.index = index

    def __repr__(self):
        return "<{}> {}".format(self.index, self.value)

    def gt_compare(self, node):
        if self.value > node.value:
            return True
        else:
            return False

class MaxHeap():

    def __init__(self, arr):
        self.heap = []
        self.heap.append(MaxHeapNode(None, 0))
        self.index = 1

        # Sort the array first, O(n log n), in creating the heap you are guaranteed a max heap
        arr = merge_sort(arr)
        for element in arr:
            self.heap.append(MaxHeapNode(element, self.index))
            self.index += 1

        # Add the parent, left/right child
        for element in self.heap:
            try:
                # Recall the indexing of the data structure is different from the actual heap!
                if element.index != 1:
                    element.parent = self.heap[element.index//2 - 1]
                element.left = self.heap[element.index*2 - 1]
                element.right = self.heap[element.index*2]
            except IndexError:
                continue

    def __repr__(self):
        return_string = ""
        for element in self.heap:
            return_string += repr(element) + '\n'
        return return_string

    def peek(self):
        return self.heap[0].value

    def swap(self, i, j):
        # Just swap the values, the connections of the tree stays the same
        self.heap[i].value, self.heap[j].value = self.heap[j].value, self.heap[i].value

    def push(self, value):
        # Place the new node at the end, set the parent
        self.heap.append(MaxHeapNode(value, self.index))
        self.heap[self.index-1].parent = self.heap[self.index//2 -1]
        if not self.heap[self.index-1].parent.left:
            self.heap[self.index-1].parent.left = self.heap[self.index-1]
        elif not self.heap[self.index-1].parent.right:
            self.heap[self.index-1].parent.right = self.heap[self.index-1]

        # Float up the value until max_heap is valid
        self.__floatup(self.index -1)
        # Increment
        self.index += 1

    def pop(self):
        # Bring the last element to the top to sink
        self.heap[0].value = self.heap[-1].value
        del self.heap[-1]
        self.index -= 1

        self.__sinkdown(0)

    def __floatup(self, index):
        # While we are not at the root and our parent is smaller than the current
        while index > 0 and self.heap[index].parent.value < self.heap[index].value:
            self.swap(index, self.heap[index].parent.index -1)
            index = self.heap[index].parent.index -1

    def __sinkdown(self, index):
        # While we are not at the end of the tree
        while index < self.index:
            try:
                # If both the children values are greater than the current
                if self.heap[index].value < self.heap[index].left.value and self.heap[index].value < self.heap[index].right.value:
                    if self.heap[index].left.value > self.heap[index].right.value:
                        self.swap(index, self.heap[index].left.index -1)
                        index = self.heap[index].left.index -1
                    else:
                        self.swap(index, self.heap[index].right.index -1)
                        index = self.heap[index].right.index -1
                # Only the left value is greater than the current
                elif self.heap[index].left.value > self.heap[index].value:
                        self.swap(index, self.heap[index].left.index -1)
                        index = self.heap[index].left.index -1
                # Only the right value is greater than the current
                elif self.heap[index].right.value > self.heap[index].value:
                        self.swap(index, self.heap[index].right.index -1)
                        index = self.heap[index].right.index -1
                # Neither of the children values is greater
                else:
                    return
            except AttributeError:
                return

########################################################################################################################

# TODO
class TreeNode():

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class BalancedTree():

    def __init__(self, root=None):
        self.root_node = root

    def set_root(self, root):
        self.root_node = root

    def insert_node(self, value):
        if not self.root_node:
            self.root_node = TreeNode(value)

        current_node = self.root_node
        while current_node:
            # Check if the insert is less than the current
            if current_node.value > value:
                # Less than the current value and left child exists already; continue traversing
                if current_node.left:
                    current_node = current_node.left
                # Empty spot, insert the new value
                else:
                    current_node.left = TreeNode(value)
            # Current insert is less than or equal to
            else:
                # Greater than the current value and right child exists already; continue traversing
                if current_node.right:
                    current_node = current_node.right
                # Empty spot, insert the new value
                else:
                    current_node.right = TreeNode(value)

        self.balance_tree()

    def get_balance(self):
        pass

    def balance_tree(self):
        pass

    def contains(self, value):
        visit = []
        visitations = 0

        visit.append(self.root)

        while visit:
            visitations += 1
            current_node = visit.pop()

            if current_node.value == value:
                return True

            if current_node.left:
                visit.append(current_node.left)
            if current_node.right:
                visit.append(current_node.right)

        print("We visited {} nodes!".format(visitations))

        return False

########################################################################################################################

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

########################################################################################################################

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

def test_node():
    node = Node('root', Node('left', Node('left.left')), Node('right'))
    assert deserialize(serialize(node)).left.left.val == 'left.left'

def input_value(arr, index, value):
    '''
    Insert the value at the specified index but insert -1 for all other indices passed that are empty.
    '''
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
    '''
    Function converts a binary tree into a string (indices correlates to the binary tree)
    '''
    tree_list = []
    traverse_list = []
    index_list = []
    index = 1
    current_node = node
    serialized_string = ""

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
    '''
    Function converts a serialized string into a binary tree (indicies correlates to the binary tree)
    '''
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
        current_node = root
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

########################################################################################################################

if __name__ == "__main__":
    e = ExerciseProblems()
    dcp = DailyCodingProblem()

    t = TreeNode(12, TreeNode(10), TreeNode(20, TreeNode(15), TreeNode(24)))
    k_sum_binary_tree(t)