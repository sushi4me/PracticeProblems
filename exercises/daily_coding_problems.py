from data_structs.common import *
from exercises.exceptions import NotImplementedException
from exercises.exercise_problems import ExerciseProblem


class DailyCodingProblem(ExerciseProblem):

    def __init__(self, diff=None):
        super(DailyCodingProblem, self).__init__()
        self.difficulty = diff

    def test(self):
        self.execute()

    def execute(self):
        raise NotImplementedException


# Regular Expression Matching
class Problem25(DailyCodingProblem):

    def __init__(self):
        super(Problem25, self).__init__()
        self.difficulty = 'Hard'

    def test(self):
        s = 'ccha'
        regex = 'c*hat'
        print("s={0}, regex={1}, result={2}".format(s, regex, self.execute(s=s, regex=regex)))

    @staticmethod
    def matches_first_char(s, r):
        return s[0] == r[0] or (r[0] == '.' and len(s) > 0)

    def execute(self, **kwargs):
        regex = kwargs['regex']
        s = kwargs['s']

        # Base cases
        if s == '' and len(regex) != 0:
            # There are no characters left in s but there are some remaining in regex, make sure that it contains no
            # more than 2 characters and * is used since it allows for length zero
            return len(regex) <= 2 and '*' in regex

        if regex == '':
            return s == ''

        # If the character is not proceeded by * then we can keep recursively verifying the next substring after
        # checking the current character
        if len(regex) == 1 or regex[1] != '*':
            if self.matches_first_char(s, regex):
                return self.execute(s=s[1:], regex=regex[1:])
            else:
                return False
        else:
            # Check to see if we match the case in which * is zero length
            if self.execute(s=s, regex=regex[2:]):
                return True

            # Make sure the first characters match then pass on the remainder of the s and everything of regex after *.
            i = 0
            while self.matches_first_char(s[i:], regex):
                if self.execute(s=s[i + 1:], regex=regex[2:]):
                    return True
                i += 1

            return False


# Removing k-Last Element from Linked List
class Problem26(DailyCodingProblem):

    def __init__(self):
        super(Problem26, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        s = SingleLinkedListNode(0)
        head = s
        s_next = SingleLinkedListNode(0)

        for i in range(1, 10):
            s_next.value = i
            s.next = s_next
            s = s.next

        self.execute(head_of_list=head, last=2)

    def execute(self, **kwargs):
        fast = slow = kwargs['head_of_list']
        n_from_last = kwargs['last']

        for i in range(n_from_last):
            fast = fast.next

        prev = None
        while fast is not None:
            prev = slow
            fast = fast.next
            slow = slow.next

        # Fast will eventually hit a None condition exiting the loop and slow will be on the Nth from last
        prev.next = slow.next


# Balanced Brackets
class Problem27(DailyCodingProblem):

    def __init__(self):
        super(Problem27, self).__init__()
        self.brackets = {')': '(', ']': '[', '}': '{'}

    def test(self):
        bracket_string = '([{}{}])'

        print(self.execute(bracket_string=bracket_string))

    def execute(self, **kwargs):
        bracket_string = kwargs['bracket_string']
        stack = []

        for char in bracket_string:
            if char in self.brackets:
                if self.brackets[char] != stack.pop():
                    return False
            else:
                stack.append(char)

        return True


# Word Wrap
class Problem28(DailyCodingProblem):

    def __init__(self):
        super(Problem28, self).__init__()

    def test(self):
        list_of_words = ['hello', 'there', 'bye']
        k = 10

        print("Each line should be less than or equal to {} characters.".format(k))
        print(self.execute(list_of_words=list_of_words, k=k))

    def execute(self, **kwargs):
        list_of_words = kwargs['list_of_words']
        k = kwargs['k']
        sorted_words = []
        current_list = []

        # Find the character count in each word
        for word in list_of_words:
            # length of all words + spaces + current word < k
            if len(''.join(current_list)) + len(current_list) - 1 + len(word) < k:
                current_list.append(word)
            # Maxed out, move to the next line
            else:
                sorted_words.append(current_list)
                current_list = [word]

        # Append the last one
        sorted_words.append(current_list)

        # Return a list which has the justified strings
        current_list = []
        for lst in sorted_words:
            current_list.append(' '.join(lst))

        return current_list


# Encoding String
class Problem29(DailyCodingProblem):

    def __init__(self):
        super(Problem29, self).__init__()

    def test(self):
        raw_string = 'AAABBBCCDA'

        encoded = self.execute(func='encode', raw_string=raw_string)
        decoded = self.execute(func='decode', encoded_string=encoded)

        print("{0} --ENCODING--> {1}".format(raw_string, encoded))
        print("{0} --DECODING--> {1}".format(encoded, decoded))

    def execute(self, **kwargs):

        def encode(keyword_args):
            raw_string = keyword_args['raw_string']
            encoded_string = ""
            current = (None, 0)

            for char in raw_string:
                if current[1] == 0:
                    current = (char, 1)
                elif current[0] == char:
                    current = (char, current[1] + 1)
                else:
                    encoded_string += str(current[1]) + current[0]
                    current = (char, 1)
            # Print the last element
            encoded_string += str(current[1]) + str(current[0])

            return encoded_string

        def decode(keyword_args):
            encoded_string = keyword_args['encoded_string']
            decoded_string = ""
            current_count = 0

            for char in encoded_string:
                if char.isdigit():
                    current_count = current_count * 10 + int(char)
                else:
                    decoded_string += char * current_count
                    current_count = 0

            return decoded_string

        funcs = {'encode': encode, 'decode': decode}
        return funcs[kwargs['func']](kwargs)


# Rain Fill-up
class Problem30(DailyCodingProblem):
    """
    Notes: The original algorithm was missing cases in which your wall does not hold any water -- all the water runs off
    the edges.  See example 1.
    """

    def __init__(self):
        super(Problem30, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        print("[1, 2, 3, 4, 3, 2, 1]={}".format(self.execute(wall=[1, 2, 3, 4, 3, 2, 1])))
        print("[1, 1, 1, 1, 1, 1, 1]={}".format(self.execute(wall=[1, 1, 1, 1, 1, 1, 1])))
        print("[1, 0, 1, 0, 1, 0, 1]={}".format(self.execute(wall=[1, 0, 1, 0, 1, 0, 1])))
        print("[3, 0, 4, 3, 5, 2, 1]={}".format(self.execute(wall=[3, 0, 4, 3, 5, 2, 1])))

    def execute(self, **kwargs):
        wall = kwargs['wall']
        total = 0

        edge = wall[0]
        # This will give you the max's index
        max_index = wall.index(max(wall))

        for value in wall[1:max_index]:
            if edge > value:
                total += edge - value
            edge = max(edge, value)

        edge = wall[-1]
        for value in wall[-2:max_index:-1]:
            if edge > value:
                total += edge - value
            edge = max(edge, value)

        return total


# Editing Distance
class Problem31(DailyCodingProblem):

    """
    Notes:
        > Dynamic Programming
    """

    def __init__(self):
        super(Problem31, self).__init__()
        self.difficulty = 'Easy'

    def test(self):
        print("cat, at={}".format(self.execute(s1='cat', s2='at')))

    def execute(self, **kwargs):
        string1 = kwargs['s1']
        string2 = kwargs['s2']

        x, y = len(string1) + 1, len(string2) + 1

        # Checking for NULL case in which we just need to insert N characters
        if len(string1) == 0 or len(string2) == 0:
            return max(len(string1), len(string2))

        edit = [[-1 for _ in range(x)] for _ in range(y)]

        # To get from NULL to N characters in another string we insert N times
        for i in range(x):
            edit[0][i] = i
        for j in range(y):
            edit[j][0] = j

        # Traverses the rows
        for j in range(1, y):
            # Traverses the columns
            for i in range(1, x):
                # If the characters match then we only need to worry about the string edit up to this point
                if string1[i - 1] == string2[j - 1]:
                    edit[j][i] = edit[j - 1][i - 1]
                # If the characters do not match then we need to find the minimum number of edits up to here plus 1
                else:
                    print(edit)
                    edit[j][i] = min(edit[j - 1][i - 1] + 1,
                                     edit[j - 1][i] + 1,
                                     edit[j][i - 1] + 1)
        return edit[-1][-1]


# Currency Exchange
class Problem32(DailyCodingProblem):

    def __init__(self):
        super(Problem32, self).__init__()

    def test(self):
        raise NotImplementedException

    def execute(self, **kwargs):
        from math import log

        conversion_table = kwargs['table']

        # This will convert small numbers less than 1.0 into large positive numbers, and large positive numbers into
        # negative numbers.
        transformed_table = [[-log(e) for e in row] for row in conversion_table]

        # Setting up variables for Bellman-Ford
        num_vertices = len(conversion_table)
        min_distance = [float('inf')] * num_vertices
        min_distance[0] = 0

        # Bellman Ford
        for i in range(num_vertices - 1):
            for j in range(num_vertices):
                for k in range(num_vertices):
                    # Check if outgoing edge has been updated due to change in vertex j
                    if min_distance[k] > min_distance[j] + transformed_table[j][k]:
                        min_distance[k] = min_distance[j] + transformed_table[j][k]

        # Run Bellman-Ford one more time to indicate negative cycle
        for i in range(num_vertices):
            for j in range(num_vertices):
                if min_distance[j] > min_distance[i] + transformed_table[i][j]:
                    return True

        return False


# Running Median
class Problem33(DailyCodingProblem):

    def __init__(self):
        super(Problem33, self).__init__()

    def test(self):
        self.execute(arr=[2, 1, 5, 7, 2, 0, 5])

    def execute(self, **kwargs):
        from heapq import heappop, heappush, _heapify_max, nlargest, nsmallest

        arr = kwargs['arr']

        greater_than_median = []
        less_than_median = []
        median = None

        for num in arr:
            # Add the number
            if median is None or num >= median:
                heappush(greater_than_median, num)
            else:
                heappush(less_than_median, num)

            # See if we need to reheapify
            if len(greater_than_median) > len(less_than_median) + 1:
                heappush(less_than_median, heappop(greater_than_median))
            elif len(less_than_median) > len(greater_than_median) + 1:
                _heapify_max(less_than_median)
                heappush(greater_than_median, heappop(less_than_median))

            # Print the running median
            if len(greater_than_median) > len(less_than_median):
                median = nsmallest(1, greater_than_median)[0]
            elif len(less_than_median) > len(greater_than_median):
                median = nlargest(1, less_than_median)[0]
            else:
                median = (nsmallest(1, greater_than_median)[0] + nlargest(1, less_than_median)[0]) / 2

            print("median={0}\tarr={1}".format(median, less_than_median + greater_than_median))


# Insert to make Palidrome
class Problem34(DailyCodingProblem):

    """
    Notes:
        > Dynamic Programming
    """

    def __init__(self):
        super(Problem34, self).__init__()

    def test(self):
        self.execute(s='race')

    def execute(self, **kwargs):
        s = kwargs['s']

        # Already a palindrome by itself
        if len(s) <= 1:
            return s

        cache = [['' for _ in range(len(s) + 1)] for _ in range(len(s) + 1)]

        for i in range(len(s)):
            cache[i][1] = s[i]

        # j denotes the number of characters in the current string
        for j in range(2, len(s) + 1):
            # i indicates the start of the current string
            for i in range(len(s) - j + 1):
                term = s[i: i + j]
                print(term)
                # Get the first and last character for comparison
                first, last = term[0], term[-1]
                # If the two characters are matching then we can just refer to the best palindrome using 2 less
                # characters and starting at one index ahead
                if first == last:
                    cache[i][j] = first + cache[i + 1][j - 2] + last
                # Otherwise use the best palindrome possible using the first/last character only with 1 less character
                # and possibly at 1 index ahead (depending on first or last)
                else:
                    one = first + cache[i + 1][j - 1] + first
                    two = last + cache[i][j - 1] + last
                    # Set based on which is shorter, otherwise use which is lexicologically first
                    if len(one) < len(two):
                        cache[i][j] = one
                    elif len(one) > len(two):
                        cache[i][j] = two
                    else:
                        cache[i][j] = min(one, two)

                print(cache)

        # Return the palindrome from the start of the string (index=0) using all characters
        return cache[0][-1]


# RGB
class Problem35(DailyCodingProblem):

    def __init__(self):
        super(Problem35, self).__init__()

    def test(self):
        print(self.execute(rgb='GBRRBGRGGGGRBBB'))

    def execute(self, **kwargs):
        rgb = list(kwargs['rgb'])
        r = g = 0
        b = len(rgb) - 1

        while g <= b:
            if rgb[g] == 'R':
                # Swap the two elements
                rgb[r], rgb[g] = rgb[g], rgb[r]
                # We swapped a 'R' to the correct section we can increment for the 'R' sub-array
                r += 1
                g += 1
            elif rgb[g] == 'B':
                # Swap the two elements
                rgb[b], rgb[g] = rgb[g], rgb[b]
                # We swapped a 'B' to the correct section, however remember that we need to check what we swapped to
                # the middle
                b -= 1
            else:
                # As we come across 'R' we will push all 'G' to the middle
                g += 1

        return ''.join(rgb)


# Second Largest Node in BST
class Problem36(DailyCodingProblem):

    """
    Notes:
        Do an inorder traverse and count
    """

    def __init__(self):
        super(Problem36, self).__init__()

    def test(self):
        node = BSTNode(0, None, BSTNode(1))
        self.execute(node=node)

    def execute(self, **kwargs):

        def inorder_traverse(node):
            # Introduced in Python3 `nonlocal` allows variables outside the scope of nested functions to be used.  In
            # Python2, static analysis by the interpreter would deem that `count` and `val` belong to `inorder_traverse`
            # and thus throw an error when we attempt to access it.
            nonlocal count, val

            if not node or count == 2:
                return

            # Traverse the right side since it is the larger value
            if node.right:
                inorder_traverse(node.right)

            # Two possibilities -- the current node is the largest and its parent is the second largest OR the current
            # node is the largest and its left subtree contains the second largest.
            count += 1
            if count == 2:
                val = node.value
                return

            if node.left:
                inorder_traverse(node.left)

        count = 0
        val = None
        inorder_traverse(kwargs['node'])
        return val


# Subset Permutations
class Problem37(DailyCodingProblem):

    def __init__(self):
        super(Problem37, self).__init__()
        self.difficulty('Easy')

    def test(self):
        print(self.execute(s=[1, 2, 3]))

    def execute(self, **kwargs):

        def power_set(s):
            if not s:
                return [[]]

            # Take out one of the values and recursively call on the remainder.  Eventually we will hit the case where
            # we have no elements, and a single element -- returns [[], [1]].  From here we just add the next element
            # to this result for the next elements -- [[], [1], [2], [1, 2]].
            result = power_set(s[1:])
            return result + [subset + [s[0]] for subset in result]

        s = kwargs['s']
        return power_set(s)


# n-Queens
class Problem38(DailyCodingProblem):

    def __init__(self):
        super(Problem38, self).__init__()
        self.difficulty = 'Hard'

    def test(self):
        n = 4
        print("For a {0}-by-{0} board, there are {1} way(s) to place {0} queens.".format(n, self.execute(queens=n)))

    def execute(self, **kwargs):

        def n_queens(n, board=[]):
            if n == len(board):
                return 1

            count = 0
            # Attempt to place the queen in each of the N columns
            for col in range(n):
                board.append(col)
                # Check for validity, if valid then recursively call to the next row
                if is_valid(board):
                    count += n_queens(n, board)
                # Board was not valid, therefore we pop the assumption and go with the queen in a different column
                board.pop()

            return count

        def is_valid(board):
            # Get the current row and column of the queen
            current_queen_row, current_queen_col = len(board) - 1, board[-1]

            # Check if it is in conflict with any of the other queens
            for row, col in enumerate(board[:-1]):
                # The diff tells us if it is in the same column
                diff = abs(current_queen_col - col)
                # If the diff is equal to the difference in the rows, then the queen is diagonally conflicted. Note here
                # that the queens will not conflict in terms of being in the same row since we are appending row-by-row.
                if diff == 0 or diff == current_queen_row - row:
                    return False

            return True

        return n_queens(kwargs['queens'])


# TODO: Conway's Game of Life
class Problem39(DailyCodingProblem):

    """
    Notes:
        Conway's Game of Life implementation
    """

    def __init__(self):
        super(Problem39, self).__init__()

    def test(self):
        pass

    def execute(self, **kwargs):
        pass


# Triples Array
class Problem40(DailyCodingProblem):

    """
    Notes:
        Time complexity of O(N) and space complexity O(1). A good way to think about this problem was to start with a
        simpler case in which we are trying to find a single value in an array which is not a pair. Thinking about the
        XOR operation that tells us if a number if equal to each other (bit-wise). If we do a bit-wise XOR with all the
        pairs in the list and then a single value, the resulting value should simply be the single value.

        Example:
            0011 ^ 0011 ^ 1001 = 0000 ^ 1001 = 1001
    """

    def __init__(self):
        super(Problem40, self).__init__()

    def test(self):
        arr = [1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        print(self.execute(arr=arr))

    def execute(self, **kwargs):
        int_size = 32
        appears_once = 0
        arr = kwargs['arr']
        result = [0 for _ in range(int_size)]

        for num in arr:
            for i in range(int_size):
                result[i] += num >> i & 1

        for i in range(int_size):
            result[i] = result[i] % 3
            if result[i] != 0 and result[i] != 1:
                return "The array should include values that appear trice and only one value which appears once."
            elif result[i] == 1:
                appears_once += 2**i

        return appears_once


# Continuous Intinerary
class Problem41(DailyCodingProblem):

    """
    Notes:
        Make use of backtracking.
    """

    def __init__(self):
        super(Problem41, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        print(self.execute(flights=[('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')], current=['YUL']))

    def execute(self, **kwargs):

        def build_itinerary(flights, current):
            if not flights:
                return current

            # Get our last destination so we can find our next flight
            last_destination = current[-1]

            # Iterate through each of the flights
            for index, (origin, destination) in enumerate(flights):
                # Flight list without one
                new_flights = flights[:index] + flights[index + 1:]
                # Append the next destination if the origin matches out current destination
                current.append(destination)
                if last_destination == origin:
                    return build_itinerary(new_flights, current)
                # In the case that the origin does not match our current destination we will backtrack and pick a new
                # flight if possible
                current.pop()

            return None

        flights = kwargs['flights']
        current = kwargs['current']
        return build_itinerary(flights, current)


# k-Sum Subarray
class Problem42(DailyCodingProblem):

    def __init__(self):
        super(Problem42, self).__init__()

    def test(self):
        arr = [12, 1, 61, 5, 9, 2]
        k = 24
        print(self.execute(arr=arr, target=k))

    def execute(self, **kwargs):
        arr = kwargs['arr']
        target = kwargs['target']
        # The question is -- Can we form _ with the first _ values?
        chart = [[None for _ in range(target + 1)] for _ in range(len(arr) + 1)]

        # How to get k=0? Just pick nothing!
        for i in range(len(arr) + 1):
            chart[i][0] = []

        # Starting from k=1 with at least length 1 array
        for j in range(1, len(arr) + 1):
            for i in range(1, target + 1):
                # Get the first element in the arr
                last = arr[j - 1]
                # If the current element is greater than the sum then just use the previous subarray since adding this
                # element will not achieve the wanted sum
                if last > i:
                    chart[j][i] = chart[j - 1][i]
                else:
                    # Means we can make the current sum without the current value
                    if chart[j - 1][i] is not None:
                        chart[j][i] = chart[j - 1][i]
                    # We might be able to make the sum using the current value, check the difference between the current
                    # sum and value
                    elif chart[j - 1][i - last] is not None:
                        chart[j][i] = chart[j - 1][i - last] + [last]
                    # We cannot make the sum regardless of the current value, set to None
                    else:
                        chart[j][i] = None
        print(chart)
        return chart[-1][-1]


# Implement a Constant-Time Stack
class Problem43(DailyCodingProblem):

    def __init__(self):
        super(Problem43, self).__init__()
        self.max_stack = []
        self.stack = []

    def test(self):
        push_values = [1, 2, 1, 3]
        pop_values = 1
        self.execute(push=push_values, pop=pop_values)

    def execute(self, **kwargs):
        push_values = kwargs['push']
        pop_values = kwargs['pop']

        for num in push_values:
            self.push(num)

        assert self.max() == 3

        for _ in range(pop_values):
            assert self.pop() == 3

        assert self.max() == 2

        print(self.stack, self.max_stack)

    def push(self, value):
        # The current max is greater than the value you are attempting to push
        if self.max_stack and self.max_stack[-1] >= value:
            self.max_stack.append(self.max_stack[-1])
        # The current max is not greater than the current value or the stack is empty
        else:
            self.max_stack.append(value)
        self.stack.append(value)

    def pop(self):
        self.max_stack.pop()
        return self.stack.pop()

    def max(self):
        return self.max_stack[-1]


# Number of Inversions in Array
class Problem44(DailyCodingProblem):

    """
    Notes:
        > Divide & Conquer
        This can be treated as a merge sort -- whenever the left value is greater than any value on the right there is
        an inversion. We end up with two sorted subarrays (left, right) in which all elements on the left should be less
        than that on the right. If we attempt to merge and end up popping a value from the left we have N inversions --
        where N is the number of elements left in the right subarray yet to be merged.
    """

    def __init__(self):
        super(Problem44, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        print(self.count_inversions(arr=[2, 4, 1, 3, 5]))

    def count_inversions(self, **kwargs):
        arr = kwargs['arr']
        count, _ = self.count_inversions_helper(arr)
        return count

    def count_inversions_helper(self, arr):
        if len(arr) <= 1:
            return 0, arr

        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        print(left, right)
        left_count, left_sorted_arr = self.count_inversions_helper(left)
        right_count, right_sorted_arr = self.count_inversions_helper(right)
        between_count, sorted_arr = self.merge_and_count(left_sorted_arr, right_sorted_arr)

        return left_count + right_count + between_count, sorted_arr

    @staticmethod
    def merge_and_count(left, right):
        count = 0
        sorted_arr = []
        i, j = 0, 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                sorted_arr.append(left[i])
                i += 1
            elif left[i] > right[j]:
                sorted_arr.append(right[j])
                count += len(left) - i
                j += 1
            sorted_arr.extend(left[i:])
            sorted_arr.extend(right[j:])

        return count, sorted_arr

    def execute(self, **kwargs):
        """
        Notes:
            This is not the optimal solution less than O(N**2) since in the worst scenario we traverse a completely
            left-sided BST which is  O(N**2).
        """
        arr = kwargs['arr']
        current_node = BSTNode(arr[0])
        head = current_node
        visit = []
        invert = []
        result = {}

        for num in arr[1:]:
            while current_node:
                # num is less than the current_node in BST
                if num < current_node.value:
                    # Visit this node and its right subtree later
                    visit.append(current_node)
                    # There are more left values to compare with
                    if current_node.left:
                        current_node = current_node.left
                    # No left value place the num
                    else:
                        current_node.left = BSTNode(num)
                        break
                # num is greater than or equal to the current_node in BST
                elif num >= current_node.value:
                    # Keep going to the right until we find a value it is less than or place it
                    if current_node.right:
                        current_node = current_node.right
                    else:
                        current_node.right = BSTNode(num)
                        break

            current_node = head

            for node in visit:
                invert += self.get_right_subtree(node)

            result[num] = invert
            visit = []
            invert = []

        return result


    @staticmethod
    def get_right_subtree(node):
        li = [node.value]
        if node.right:
            visit = [node.right]
        else:
            visit = []

        while visit:
            li += [visit[0].value]
            if visit[0].left:
                li.append(visit[0].left)
            if visit[0].right:
                li.append(visit[0].right)
            visit.pop(0)

        return li


# Implement rand7 using rand5
class Problem45(DailyCodingProblem):

    """
    Notes:
        Get a range that includes an even number of multiple of 7, we can truncate some values to make it even. For
        example, if we can get a [1, 25] range then we can truncate [22, 25] if they appear and only return when we
        have [1, 21] to use modulo 7 on.
    """

    def __init__(self):
        super(Problem45, self).__init__()

    def test(self):
        self.test_rand()

    @staticmethod
    def test_rand():
        k = 5

        for i in range(k):
            for j in range(k):
                print(3 * i + j + 1)

    def execute(self, **kwargs):

        def rand5():
            from random import uniform
            return uniform(1, 5)

        # (5 - 25) + (1 - 5) - 5
        # This yields a number from 1 - 25
        num = 5 * rand5() + rand5() - 5

        while num > 21:
            num = 5 * rand5() + rand5() - 5

        return num % 7


# Longest Palindromic Substring
class Problem46(DailyCodingProblem):

    """
    Notes:
        > Dynamic Programming
    """

    def __init__(self):
        super(Problem46, self).__init__()
        self.difficulty = 'Hard'

    def test(self):
        s = 'anabbbbbbbb'
        print("s={0}, longest_palindrome={1}".format(s, self.execute(s=s)))

    def execute(self, **kwargs):
        s = kwargs['s']
        n = len(s)
        A = [[1 if i == j else 0 for i in range(n)] for j in range(n)]

        for j in range(n):
            for i in reversed(range(j + 1)):
                # Single character is considered a palindrome of length 1
                if i == j:
                    A[i][j] = 1
                # The two characters at the ends are equal, therefore we can just check the previous
                elif s[i] == s[j]:
                    # Our palindrome is as long the best palindrome we can get without the two end characters
                    A[i][j] = 2 + A[i + 1][j - 1]
                # Otherwise we do not have a palindrome and the best we can do is the best from not having either or
                # end characters
                else:
                    # No last character, no first character, no first and last character
                    A[i][j] = max(A[i][j - 1], A[i + 1][j], A[i + 1][j - 1])

        return A[0][-1]


# Maximum Stock Profit
class Problem47(DailyCodingProblem):

    def __init__(self):
        super(Problem47, self).__init__()
        self.difficulty = 'Easy'

    def test(self):
        stocks = [999, 2, 10, 99, 1, 7, 100, 50, 22, 6]
        print(self.execute(stocks=stocks))

    @staticmethod
    def search(arr, op):
        ret = None

        if len(arr) < 1:
            return None, None

        for index, num in enumerate(arr):
            if ret is None or op(ret[1], num):
                ret = (index, num)

        return ret

    def execute(self, **kwargs):
        from operator import lt, gt
        stocks = kwargs['stocks']

        # Search for the smallest and largest elements
        l_index, largest = self.search(stocks, lt)
        s_index, smallest = self.search(stocks, gt)

        # Search before/after for the smallest/largest to maximize difference (profit)
        _, left_smallest = self.search(stocks[:l_index], gt)
        _, right_largest = self.search(stocks[s_index + 1:], lt)

        # Do a small check here in one of the cases in which the maximums are at the edges
        if left_smallest is None:
            return right_largest - smallest
        if right_largest is None:
            return largest - left_smallest

        # Return the larger profit of the two
        ret1 = largest - left_smallest
        ret2 = right_largest - smallest

        return ret1 if ret1 > ret2 else ret2


# Reproduce Tree given Pre/In-order
class Problem48(DailyCodingProblem):

    """
    Notes:
        > Recursive
    """

    def __init__(self):
        super(Problem48, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        pass

    def execute(self, **kwargs):
        inorder = kwargs['inorder']
        preorder = kwargs['preorder']

        if len(preorder) == 0 and len(inorder) == 0:
            return None
        elif len(preorder) == len(inorder) == 1:
            return TreeNode(value=preorder[0])

        root = TreeNode(value=preorder[0])
        mid_index = inorder.index(root.value)
        """
        Simply extract the root value and pass the corresponding sides to the function recursively to build each side
            inorder:  | left | root | right |
            preorder: | root | left | right |
        """
        root.left = self.execute(inorder=inorder[:mid_index], preorder=preorder[1: mid_index + 1])
        root.right = self.execute(inorder=inorder[mid_index + 1:], preorder=preorder[mid_index + 1:])


# Largest Continuous Subarray
class Problem49(DailyCodingProblem):

    def __init__(self):
        super(Problem49, self).__init__()
        self.difficulty = 'Medium'

    def test(self):
        arr = [1, 2, 3, 4, -10, 99, -1, 1000]
        print("arr={0}, best={1}".format(arr, self.execute(arr=arr)))

    def execute(self, **kwargs):
        arr = kwargs['arr']
        best, total = 0, 0

        for num in arr:
            if total + num <= 0:
                if total > best:
                    best = total
                total = 0

            else:
                total += num

        if total > best:
            return total

        return best


# Operation Tree
class Problem50(DailyCodingProblem):

    def __init__(self):
        super(Problem50, self).__init__()
        self.difficulty = 'Easy'

    def test(self):
        root = TreeNode(value='+',
                        left=TreeNode(value='*',
                                      left=TreeNode(value=2),
                                      right=TreeNode(value=3)),
                        right=TreeNode(value=5))
        print("Result: {}".format(self.execute(root=root)))

    def execute(self, **kwargs):
        from operator import add, sub, mul, truediv, mod, xor

        def get_operator(op_string):
            try:
                return {
                    '+': add,
                    '-': sub,
                    '*': mul,
                    '/': truediv,
                    '%': mod,
                    '^': xor,
                }[op_string]
            except KeyError:
                return False

        root = kwargs['root']
        op = get_operator(root.value)

        if op:
            return op(self.execute(root=root.left), self.execute(root=root.right))
        else:
            return root.value


class Problem51(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self):
        pass

    def execute(self, **kwargs):
        from random import randint
        cards = kwargs['cards']
        n = len(cards)

        for i in range(n - 1):
            j = randint(i, n - 1)
            cards[i], cards[j] = cards[j], cards[i]
