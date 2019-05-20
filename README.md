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

This project does not have any particular training data. The optimal path traversal is a basic reinforcement learning problem to solve by creating a path and using the principles of Q Learning to update the Q matrix to maximize rewards. The algorith employs exploratoion and exploitation to try to find an optimal path. Once the training is complete, a "best path" can be found by taking optimal moves at each state from the values in the Q matrix.

## References

* [Reinforcement Learning Tutorial](https://visualstudiomagazine.com/articles/2018/10/18/q-learning-with-python.aspx) - Used as guide

## Libraries Used

* numpy
* matplotlib
