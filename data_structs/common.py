from exercises.exercise_problems import ExerciseProblems


class MaxHeapNode:

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


class MaxHeap:

    def __init__(self, arr):
        self.heap = []
        self.heap.append(MaxHeapNode(None, 0))
        self.index = 1

        # Sort the array first, O(n log n), in creating the heap you are guaranteed a max heap
        e = ExerciseProblems()
        arr = e.merge_sort(arr)
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


# TODO
class BalancedTree:

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

        visit.append(self.root_node)

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

        left_count, left_unival = TreeNode.count_unival_recursive(node.left)

        right_count, right_unival = TreeNode.count_unival_recursive(node.right)

        total_count = left_count + right_count

        if left_unival and right_unival:
            if node.left is not None and node.value != node.left.value:
                return total_count, False
            if node.right is not None and node.value != node.right.value:
                return total_count, False
            return total_count + 1, True
        return total_count, False

    def convert_tree(self):
        """
        Converts a tree into a dictionary with indexes as their keys
        """
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


# Class for Problem 11: Autocomplete [Hard]
class Trie(object):

    def __init__(self):
        self.values = {}

    def insert(self, prefix):
        temp = self.values

        for char in prefix:
            if char not in temp:
                temp[char] = {}
            temp = temp[char]

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


# Class for Problem 18: Maximum Subarray [Hard]
class Log(object):

    def __init__(self, capacity):
        self.log_list = []
        self.current = 0
        self.capacity = capacity

    def record(self, order_id):
        if len(self.log_list) >= self.capacity:
            self.log_list[self.current] = order_id
        else:
            self.log_list.append(order_id)

        # Modulo with capacity will cause the list to be circular, a maxed out list will replace the oldest log
        self.current = (self.current + 1) % self.capacity

    def get_last(self, i):
        # Returns the i-th last element, remember indexed 0
        return self.log_list[self.current - i]
