from Exercises.daily_coding_problems import DailyCodingProblem
from Exercises.exercise_problems import ExerciseProblems

if __name__ == "__main__":
    e = ExerciseProblems()
    dcp = DailyCodingProblem()

    assert(dcp.parse_directory("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext") == 20)
    