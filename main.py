from exercises.daily_coding_problems import *
from exercises.book_problems import *


if __name__ == "__main__":

    p = Book_20_1()
    p.test()

    def _printing_portion():
        list_of_lists = [['hi'], ['1', '2'], ['a', 'b', 'c']]
        print_list = []
        curr = 1
        for lst in list_of_lists:
            num_of_elements = len(lst)
            if len(print_list) == 0:
                print_list.append(lst)
            else:
                print_list = print_list * num_of_elements
                for index, element in enumerate(lst):
                    for i in range(index * curr, len(print_list), curr * len(lst)):
                        for j in range(i, i + curr):
                            print_list[j] = [print_list[j][0] + element]

                curr *= len(lst)

        print(print_list)
