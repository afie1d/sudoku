import sys
import math

###################
#   CSSE 413 AI
#   Sudoku CSP Solution
#   Adam Field
#   03/08/2024
###################

class Sudoku:
    def __init__(self, filename):
        self.board_size = 0
        self.partition_size = 0
        self.vals = []

        self.read_file(filename)
        self.solve(filename)

    def read_file(self, filename):
        try:
            with open(filename, 'r') as file:
                self.board_size = int(file.readline())
                self.partition_size = int(math.sqrt(self.board_size))
                print(f"Board size: {self.board_size}x{self.board_size}")
                print("Input:")
                for i, line in enumerate(file): # starts after the first line
                    row = list(map(int, line.split())) # Split by whitespace, convert to an integer, and store it in a list
                    if len(row) != self.board_size:
                        raise RuntimeError(f"Incorrect Number of inputs.\n {row}")
                    for j, num in enumerate(row):
                        if num == 0:
                            break
                    print(' '.join(f'{num:3d}' for num in row))  # Print each number in the row formatted to be 3 digits wide for alignment.
                    self.vals.append(row)  # represents the Sudoku board.

        except FileNotFoundError:
            print(f'Input file not found: {filename}')
            sys.exit(1)

    def solve(self, filename):
        vars = self.getVars()
        solved = self.backtrack_solve(0, vars)

        solutionFile = filename[:-4] + "Solution.txt"
        f = open(solutionFile, "w")

        if not solved:
            print("No solution found.\n")
            f.write("-1")
            f.close
            return False
        print("\nOutput\n")
        for row in self.vals:
            print(' '.join(f"{num:3d}" for num in row))
            f.write(' '.join(f"{num:3d}" for num in row))
            f.write("\n")
            f.close
        return True

    # Note that curIndex is the Variable index, not board index
    def backtrack_solve(self, curIndex, vars):
        var = vars[curIndex]
    
        for i in range(self.board_size):
            var.value = i + 1
            if self.valid(var): 
                self.vals[var.index // self.board_size][var.index % self.board_size] = var.value
                if curIndex == len(vars) - 1: return True
                if self.backtrack_solve(curIndex + 1, vars): return True
                var.value = 0
                self.vals[var.index // self.board_size][var.index % self.board_size] = 0

        return False
    
    def getVars(self):
        vars = []
        for i, row in enumerate(self.vals):
            for j, num in enumerate(row):
                if num == 0:
                    vars.append(Variable(i * self.board_size + j)) # Add new variable with appropriate board index
        return vars
    
    def valid(self, var):
        if var.value == 0:
            return False
        
        # Check row
        rowNum = var.index // self.board_size
        for i, num in enumerate(self.vals[rowNum]):
            if num == var.value and i != var.index % self.board_size:
                return False
        
        # Check column
        col = var.index % self.board_size
        for j, row in enumerate(self.vals):
            if row[col] == var.value and j != var.index // self.board_size:
                return False
        
        # Check box
        xBox = col // self.partition_size
        yBox = rowNum // self.partition_size
        for i in range(self.partition_size):
            for j in range(self.partition_size):
                y = yBox * self.partition_size + i 
                x = xBox * self.partition_size + j
                if self.vals[y][x] == var.value and var.index != self.board_size * y + x:
                    return False

        return True


class Variable:
    def __init__(self, index):
        self.index = index
        self.value = 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sudoku.py <filename>")
        sys.exit(1)
    Sudoku(sys.argv[1])
