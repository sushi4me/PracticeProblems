from collections import deque
from data_structs.common import TreeNode, Trie, Log


class DailyCodingProblem:

    # From Daily Coding Problem

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
        # can be found using a power of 2.  We subtract 1 in order to account for over counting in the case they are all
        # "off" -- entire string is treated as single digit codes.
        return 2**possible_double + 2**possible_double_offset - 1

    # Problem 8: Unival tree [Easy]
    @staticmethod
    def is_unival_test_case():
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

        suffixes = t.elements(s)

        return [s + w for w in suffixes]

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

        for i in range(0, len(s)):
            count[ord(s[i]) - ord('a')] += 1
            end_index += 1

            # Verify our current window, keep "popping" until is_valid returns True
            while not is_valid(count, k):
                count[ord(s[start_index]) - ord('a')] -= 1
                start_index += 1

            if end_index - start_index + 1 - 1 > max_length:
                max_length = end_index - start_index + 1 - 1

        return max_length

    # TODO: Problem 16: Implementing Log API [Easy]
    @staticmethod
    def log_api():
        log = Log(10)
        for i in range(10):
            log.record(i)
        log.get_last(1)

    # Problem 17: Parse Directory [Hard]
    @staticmethod
    def parse_directory(dir_string):

        def set_list(li, index, value):
            try:
                while index + 1 < len(li):
                    li.pop()
                li[index] = value
            except IndexError:
                for _ in range(index - len(li) + 1):
                    li.append(None)
                li[index] = value

        split_dir = dir_string.split('\n')
        curr_max = 0
        curr_tabs = 0
        temp_arr = []
        max_str = None
        
        for part in split_dir:
            tabs = part.count('\t')
            if curr_tabs >= tabs:
                # Find string length of the temp_arr
                temp_arr_str = '/'.join(str(e).replace('\t', '') for e in temp_arr)
                if len(temp_arr_str) > curr_max:
                    curr_max = len(temp_arr_str)
                    max_str = temp_arr_str
            else:
                curr_tabs += 1

            curr_tabs = tabs
            set_list(temp_arr, tabs, part)

        temp_arr_str = '/'.join(str(e).replace('\t', '') for e in temp_arr)
        if len(temp_arr_str) > curr_max:
            curr_max = len(temp_arr_str)
            max_str = temp_arr_str

        return curr_max

    # Problem 18: Maximum Subarray [Hard]
    @staticmethod
    def max_subarray(arr, k):
        lst = deque()

        def validate_window(lst):
            pass

        # For the first k-elements - we are only interested in the largest elements
        for i in range(k):
            # While the new element is greater than the last in the list, remove it
            while lst and arr[i] >= lst[-1]:
                lst.pop()
            lst.append(i)

        for i in range(k, len(arr)):
            continue

        return lst

    # Problem 19:
    @staticmethod
    def paint_houses(arr):
        pass


class BookProblem:

    @staticmethod
    def reconstruct(preorder, inorder):
        if not preorder and not inorder:
            return None

        if len(preorder) == len(inorder) == 1:
            return preorder[0]

        root = preorder[0]
        root_i = inorder.index(root)
        root.left = BookProblem.reconstruct(preorder[1:1 + root_i], inorder[0:root_i])
        root.right = BookProblem.reconstruct(preorder[1 + root_i:], inorder[root_i + 1:])
