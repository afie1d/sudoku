# sudoku
This project examines two approaches to solving sudoku puzzles: 
- Backtracking (brute force)
- Convolutional Neural Network

# Results


# Comment on Backtracking Algorithm
In what sense is this program intelligent?

For humans, the game of Sudoku requires immense planning, trial, and error. The same is true for a computer. For both 
parties, the strategy they employ can significantly reduce or prolong the time it takes to solve a given puzzle. However,
it is clear that computers are equipped to solve any Sudoku board much faster than a human, no matter which strategy 
each player uses. It is thus irrelevant to compare this program’s level of Sudoku-solving intelligence to that of a 
human on the basis of speed and accuracy. The more interesting comparison is that of their strategies. 

This computer program fulfills the most basic definitions of intelligence due to the way it solves the puzzle. It utilizes
its memory to make a series of decisions that make progress toward a solution. Specifically, the program stores the state 
of the board, along with the spaces on the board that require input. It then iterates through the board, making comparisons 
and validations to determine its next insertion. At a high level, this is what a human typically does when solving a Sudoku 
puzzle. They will maintain knowledge of the state of the board while analyzing its contents to plan their next move. In this
way, the program clearly resembles human-like intelligence. I contest, however, that the more interesting question is the 
degree to which it resembles human-like intelligence. 

The backtracking algorithm used in this program is highly advanced when compared to a brute-force approach. Instead of trying 
every possible combination of inputs to find a solution, it reasons upon its data to inform its decisions, thus avoiding useless
computations. To me, this scores some intelligence points. However, when compared with a human-like Sudoku strategy, the program
has a sever limitation: scope.  A human can see an entire board at once when making decisions, but this program is limited to a
fixed list of variables from which it cannot deviate. To my mind, this gives an advantage to the human for being able to spot 
problems ahead-of-time. Such a dynamic analysis of the board seems synonymous with intelligence, while the computer program’s 
static approach smells of hard-coded, lifeless instructions. 

This program is certainly more intelligent than other approaches. And it has basic properties of intelligence. However, when compared
with human strategy its limitations reveal its true character. Is it intelligent? Sure. But I’m more impressed by a human that can 
adapt its strategy on-the-fly.
