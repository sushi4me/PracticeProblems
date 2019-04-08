from data_structs.common import *
from exercises.exceptions import NotImplementedException
from exercises.exercise_problems import ExerciseProblem


class DailyCodingProblem(ExerciseProblem):

    def __init__(self, diff=None):
        self.difficulty = diff

    def test(self):
        self.execute()

    def execute(self):
        raise NotImplementedException


class Problem26(DailyCodingProblem):

    def __init__(self):
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


class Problem27(DailyCodingProblem):

    def __init__(self):
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


class Problem28(DailyCodingProblem):

    def __init__(self):
        pass

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


class Problem29(DailyCodingProblem):

    def __init__(self):
        pass

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
            encoded_string += str(current[1]) + current[0]

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


class Problem30(DailyCodingProblem):
    """
    Notes: The original algorithm was missing cases in which your wall does not hold any water -- all the water runs off
    the edges.  See example 1.
    """

    def __init__(self):
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


class Problem31(DailyCodingProblem):

    """
    Notes:
        This is a dynamic programming problem.
    """

    def __init__(self):
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


class Problem32(DailyCodingProblem):

    def __init__(self):
        pass

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


class Problem33(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self):
        self.execute(arr=[2, 1, 5, 7, 2, 0, 5])

    def execute(self, **kwargs):
        from heapq import heapify, heappop, heappush, _heapify_max, nlargest, nsmallest

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


class Problem34(DailyCodingProblem):

    """
    Notes:
        This is a dynamic programming problem.
    """

    def __init__(self):
        pass

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

        for j in range(2, len(s) + 1):
            for i in range(len(s) - j + 1):
                term = s[i: i + j]
                print(term)
                first, last = term[0], term[-1]
                if first == last:
                    cache[i][j] = first + cache[i + 1][j - 2] + last
                else:
                    one = first + cache[i + 1][j - 1] + first
                    two = last + cache[i][j - 1] + last
                    if len(one) < len(two):
                        cache[i][j] = one
                    elif len(one) > len(two):
                        cache[i][j] = two
                    else:
                        cache[i][j] = min(one, two)

                print(cache)


class Problem35(DailyCodingProblem):

    def __init__(self):
        pass

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


class Problem36(DailyCodingProblem):

    """
    Notes:
        Do an inorder traverse and count
    """

    def __init__(self):
        pass

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


class Problem37(DailyCodingProblem):

    def __init__(self):
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


class Problem38(DailyCodingProblem):

    def __init__(self):
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


class Problem39(DailyCodingProblem):

    """
    Notes:
        Conway's Game of Life implementation
    """

    def __init__(self):
        pass

    def test(self):
        pass

    def execute(self, **kwargs):
        pass


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
        pass

    def test(self):
        arr = [1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        print(self.execute(arr=arr))

    def execute(self, **kwargs):
        INT_SIZE = 32
        appears_once = 0
        arr = kwargs['arr']
        result = [0 for _ in range(INT_SIZE)]

        for num in arr:
            for i in range(INT_SIZE):
                result[i] += num >> i & 1

        for i in range(INT_SIZE):
            result[i] = result[i] % 3
            if result[i] != 0 and result[i] != 1:
                return "The array should include values that appear trice and only one value which appears once."
            elif result[i] == 1:
                appears_once += 2**i

        return appears_once


class Problem41(DailyCodingProblem):

    """
    Notes:
        Make use of backtracking.
    """

    def __init__(self):
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
