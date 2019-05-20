# EECS_738_Project_4

This is Andre Kurait and Surabhi Khachar's project 4 for EECS 738. The project requirements are as follows:

1. Set up a new git repository in your GitHub account
2. Think up a map-like environment with treasure, obstacles
and opponents
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how reinforcement learning can be
used to find treasure efficiently while avoiding obstacles
and opponents
5. Build one or more reinforcement policies to model
situational assessments, actions and rewards
programmatically
6. Document your process and results
7. Commit your source code, documentation and other
supporting files to the git repository in GitHub

## Project code

The code for this is written in a Python Jupyter notebook. The output of the code can be seen in the treasureHunting.ipynb file. To run, download the repository and use the Anaconda environemnt for running Jupyter notebooks.

### Reinforcement Learning

This project does not have any particular training data. The maze solver is a basic reinforcement learning problem to solve by creating a maze and using the algorithm designed to find a traversal matrix that avoids obstacles and opponents. The program trains by continuously starting at a random state in the designed maze. From each random state, the program tries to identify all the next states. From all next states, it also identifies all possible next states. The algorithm will continue this until it has reached the state that is the goal state.

## References

* [Reinforcement Learning Tutorial](https://visualstudiomagazine.com/articles/2018/10/18/q-learning-with-python.aspx) - Used as guide

## Libraries Used

* numpy
* matplotlib
