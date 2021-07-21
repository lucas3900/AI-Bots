"""
Author: Lucas Barusek
Date: 02/21/2021

Support code for vacuum.py

Description: Extends VacuumEnvironment to return a string 
representation of the environment, and see if we've met our goal.
Additionally create an agent that randomly traverses environments
Finally hard code in several solvable environment for the agent to solve
"""

import vacuum_environment 
import matplotlib.pyplot as plt
from random import randrange, choice

class VacuumEnvironment375(vacuum_environment.VacuumEnvironment):

    def __str__(self):
        """ Prints string representation of the current state of the environment
        $ represents the outside barrier of the environment
        ' ' represents an empty space
        D represent a dirty space
        A represents the location of the agent 
        X represents an obstacle """

        # add the top row of the barrier to the output
        output = '$' * self.cols + "$$\n"
        for row in range(self.rows):

            # add the lhs of the barrier
            output += '$'
            for col in range(self.cols):

                # add the first letter of the thing occupying the current space
                # or a space character if the current space is empty
                #replace obstacle '0' with letter 'X' for readability
                type_of_space = self[(row, col)][0] if self[(row, col)][0] != 'O' else 'X'
                output += type_of_space if type_of_space != 'E' else ' '

            # add the rhs of the barrier
            output += "$\n"

        # return the output with the bottom of the barrier
        return output + '$' * self.cols + "$$\n"


    def __eq__(self, otherEnv):
        if self.rows != otherEnv.rows and self.cols != otherEnv.cols:
            return False
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self[(row, col)] != otherEnv[(row, col)]:
                    return False

        return True
        
    def __hash__(self):
        return hash(tuple([self.rows, self.cols]+[item for sublist in self.grid for item in sublist]))

    def all_clean(self):
        """ Returns true iff there are no dirty spaces in the environment """

        for row in range(self.rows):
            for col in range(self.cols):
                if self[(row, col)] == "Dirt":
                    return False

        return True


def random_agent(vacuum_env):
    """ Given a Vacuum environment, move the agent in random directions
    until there are no more dirty space, and print out the state of the environment
    after each step. Also returns the num_steps """

    num_steps = 0

    # randomly select a direction and move in that direction (if possible)
    while not vacuum_env.all_clean():
        print(vacuum_env)
        vacuum_env.move_agent(choice(['UP', 'DOWN', 'LEFT', 'RIGHT']))
        num_steps += 1
    
    # print end configuration, num_Steps, and return the number of steps
    print(vacuum_env)
    print(num_steps)
    return num_steps


def make_vacuum_environment(k):
    """ Given an index k, returns the vacuum envirnment that corresponds to the
    given index (a smaller index will give you a smaller environment 
    simple 8x8 is index 6 """

    if k == 0: return VacuumEnvironment375(1, 1, (0, 0))
    elif k == 1: return VacuumEnvironment375(2, 2, (0, 0), [(0, 0), (0, 1), (1, 0)])
    elif k == 2: return VacuumEnvironment375(
        5, 2, (2, 1), [(1,0), (3, 1), (4, 1)], [(1,1), (3, 0)]
    )
    elif k == 3: return VacuumEnvironment375(7, 3, (0,0), 
        [(0,0), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), (4, 0),\
         (4, 2), (5, 0), (5, 2), (6, 0), (6, 1), (6, 2)], 
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    )
    elif k == 4: return VacuumEnvironment375(6, 7, (3, 0),
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 0), (4, 6),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
        [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (3 , 5), (2, 5), (1, 5)]
    )
    elif k == 5: return VacuumEnvironment375(7, 6, (6, 5), 
        [(1,0)], [(1, 1), (2, 2), (3, 3)]
    )
    elif k == 6: return VacuumEnvironment375(8, 8, (3, 3))
    elif k == 7: return VacuumEnvironment375(3, 30, (0, 0), 
        [(i, j) for i in range(3) for j in range(28)],
        [(0, 1), (1, 1), (2, 3), (1, 3), (0, 5), (1, 5), (2, 7), (1, 7),\
         (0, 9), (1, 9), (2, 11), (1, 11), (0, 13), (1, 13), (2, 15),\
         (1, 15), (0, 17), (1, 17), (2, 19), (1, 19), (0, 21), (1, 21), 
         (2, 23), (1, 23), (0, 25), (1, 25), (2, 27), (1, 27)]
    )
    elif k == 8: return VacuumEnvironment375(10, 10, (1, 1), 
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)], 
        [(0, 3), (1, 3), (2, 3), (3, 3), (3, 0), (3, 1), (3, 4), (3, 5), \
         (3, 6), (3, 8), (3, 9)]
    )
    elif k == 9: return VacuumEnvironment375(10, 16, (0,14), [],
        [(0,0),(1,0),(2,0),(0,6),(0,7),(0,8),(0,9),(0,15),(1,15),(2,15),\
        (2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,2),(3,4),(3,5),(3,7),(3,8),(3,10),(3,11),(3,13),(3,14),(3,15),\
        (4,0),(4,1),(4,2),(4,4),(4,5),(4,7),(4,8),(4,10),(4,11),(4,13),(4,14),(4,15),\
        (5,0),(5,1),(5,2),(5,4),(5,5),(5,7),(5,8),(5,10),(5,11),(5,13),(5,14),(5,15),\
        (6,0),(6,1),(6,2),(6,4),(6,5),(6,7),(6,8),(6,10),(6,11),(6,13),(6,14),(6,15),\
        (7,7),(7,8),(8,7),(8,8),(9,7),(9,8)])
    elif k == 10: return VacuumEnvironment375(15, 15, (0, 0),
        [(i, j) for i in range(15) for j in range(15) if (i, j) != (14, 14) and (i, j) != (0, 14)],
        [(i, 2) for i in range(14) if i != 7] + [(1, j) for j in range(4, 15)] +\
        [(4, j) for j in range(2, 15) if j != 10] 
    )
    elif k == 11: return VacuumEnvironment375(10, 16, (0,14),
        [(0,1),(0,2),(0,3),(0,4),(0,5),(0,10),(0,11),(0,12),(0,13),\
        (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),\
        (2,1),(2,2),(2,3),(2,4),(2,5),(2,10),(2,11),(2,12),(2,13),(2,14),\
        (3,3),(3,12),(4,3),(4,6),(4,9),(4,12),\
        (5,3),(5,6),(5,9),(5,12),(6,3),(6,6),(6,9),(6,12),\
        (7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),(7,15),\
        (8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(8,15),\
        (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,9),(9,10),(9,11),(9,12),(9,13),(9,14),(9,15),],\
        [(0,0),(1,0),(2,0),(0,6),(0,7),(0,8),(0,9),(0,15),(1,15),(2,15),\
        (2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,2),(3,4),(3,5),(3,7),(3,8),(3,10),(3,11),(3,13),(3,14),(3,15),\
        (4,0),(4,1),(4,2),(4,4),(4,5),(4,7),(4,8),(4,10),(4,11),(4,13),(4,14),(4,15),\
        (5,0),(5,1),(5,2),(5,4),(5,5),(5,7),(5,8),(5,10),(5,11),(5,13),(5,14),(5,15),\
        (6,0),(6,1),(6,2),(6,4),(6,5),(6,7),(6,8),(6,10),(6,11),(6,13),(6,14),(6,15),\
        (7,7),(7,8),(8,7),(8,8),(9,7),(9,8)])
    else:
        raise IndexError(f"No environment for index {k}. Valid environments are 0 - 10")


def main():
    """ Main driver of the program. For statistic aggregation only """

    with open('env.txt', 'w') as env:
        for i in range(12):
            env.write(f'Environment {i}\n')
            env.write(str((make_vacuum_environment(i))))
            env.write('\n\n')
    
    # steps = [random_agent(make_vacuum_environment(6)) for _ in range(10000)]
    # plt.hist(steps, color='blue', edgecolor='black', bins=100)
    # plt.title('Histogram of Steps Vacuum Takes to Clean')
    # plt.xlabel('Steps Taken')
    # plt.ylabel('Frequency')
    # pltText = f"Max: {max(steps)} Steps\nMin: {min(steps)} Steps\nAverage: {sum(steps) / len(steps):.0f} Steps"
    # plt.text(8000, 400, pltText)
    # plt.show()


if __name__ == "__main__":
    main()