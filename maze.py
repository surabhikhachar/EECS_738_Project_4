# maze.py
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

NORMAL = -.1
TREASURE = 1
OBSTICLE = .5
OPPONENT = -1

def transitionToNextState(s,move,width):
  if move == 0:
    return s - width
  elif move == 1:
    return s + 1
  elif move == 2:
    return s + width
  elif move == 3:
    return s - 1
  else:
    raise Exception

def get_poss_moves(s, M):
  poss_moves = []
  valid_moves = []
  x, y = stateToXY(s, len(M[0]))
  if y > 0:
    valid_moves.append(0)
  if x < len(M[0]) - 1:
    valid_moves.append(1)
  if y < len(M) - 1:
    valid_moves.append(2)
  if x > 0:
    valid_moves.append(3)
  for move in valid_moves:
      poss_state = transitionToNextState(s,move,len(M[0]))
      x,y = stateToXY(poss_state, len(M[0]))
      if M[y][x] != OBSTICLE: poss_moves.append(move)
  return poss_moves 
def get_best_next_move(s, M, Q):
  poss_next_moves = get_poss_moves(s, M)
  if len(poss_next_moves) == 0:
      return None
  max_m = poss_next_moves[0]
  max_q = Q[s][max_m]
    
  for move in poss_next_moves[1::]:
    if(Q[s][move] > max_q):
      max_q, max_m = Q[s][move], move
  return max_m

def get_rnd_next_move(s, M, Q):
  poss_next_moves = get_poss_moves(s, M)
  if len(poss_next_moves) == 0:
      return None
  if np.random.rand() <= .8:
    return get_best_next_move(s, M, Q)
  else:
    next_move = \
      poss_next_moves[np.random.randint(0,\
      len(poss_next_moves))]
    return next_move

def stateToXY(s,width):
  return s % width, s // width
def XYToState(x,y,width):
  return y*width+x

def train(M, Q, gamma, lrn_rate, goal, ns, max_epochs, startingState, max_steps):
  for _ in range(0,max_epochs):
    M_copy = np.copy(M)
    curr_s = 0
    # curr_s = np.random.randint(0,ns)
    steps = 0
    x,y = stateToXY(curr_s, len(M_copy[0]))
    points = M_copy[y][x]
    M_copy[y][x] = OBSTICLE
    while(True):
      next_m = get_rnd_next_move(curr_s, M_copy, Q)
      if next_m is None:
        break
      next_s = transitionToNextState(curr_s,next_m,len(M[0]))
      x,y = stateToXY(next_s, len(M_copy[0]))
      next_r = M_copy[y][x]
      points += next_r
      
      M_copy[y][x] = OBSTICLE

      bestnextnextMove = get_best_next_move(next_s, M_copy, Q)
      max_Q = -1
      if bestnextnextMove is not None:
        max_Q = Q[next_s][bestnextnextMove]
       
      # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
      Q[curr_s][next_m] = Q[curr_s][next_m] + (lrn_rate) * \
                          ( next_r + (gamma * max_Q) - Q[curr_s][next_m] )
      curr_s = next_s
      steps += 1
      if curr_s == goal: break
      if steps == max_steps: break
def walk(start, goal, Q, M, max_steps):
  # cmap = mpl.colors.ListedColormap(['grey', 'gold', 'black', 'red'])
  M_copy = np.copy(M)

  cmap = mpl.colors.ListedColormap(['grey', 'gold', 'black', 'red'])

  M[abs(M - NORMAL) < .01] = -10
  M[abs(M - TREASURE) < .01] = -9
  M[abs(M - OBSTICLE) < .01] = -8
  M[abs(M - OPPONENT) < .01] = -7

  M[M is -10] = 0
  M[M is -9] = 1
  M[M is -8] = 2
  M[M is -7] = 3

  plt.pcolormesh(M, cmap=cmap)

  white_patch = mpl.patches.Patch(color='grey', label='Normal Ground')
  gold_patch = mpl.patches.Patch(color='gold', label='Treasure')
  black_patch = mpl.patches.Patch(color='black', label='Obsticle')
  red_patch = mpl.patches.Patch(color='red', label='Opponent')
  plt.subplots_adjust(right=0.7)
  plt.legend(handles=[white_patch, gold_patch, black_patch, red_patch] ,loc='upper left', bbox_to_anchor=(1.04,1), borderaxespad=0)


  steps = 1
  curr = start
  curr_x, curr_y = stateToXY(curr, len(M[0]))
  points = M_copy[curr_y][curr_x]
  M_copy[curr_y][curr_x] = OBSTICLE
  print(str((curr_x,curr_y)) + "->", end="")
  path = [[.5,.5]]
  while curr != goal and steps < max_steps:
    next_m = get_best_next_move(curr, M_copy, Q)
    if next_m is None:
      break
    next = transitionToNextState(curr,next_m,len(M[0]))
    curr_x, curr_y = stateToXY(curr, len(M[0]))
    x, y = stateToXY(next, len(M_copy[0]))
    path.append([x+.5,y+.5])
    # print(next)
    print(str((x,y)) + "->", end="")
    points += M_copy[y,x]
    M_copy[y][x] = OBSTICLE
    
    curr = next
    steps += 1
  print("\ndone. The optimal path has point value of ", points)
  data = np.array(path)
  plt.plot(*data.T, color='red')

  plt.axes().set_aspect('equal') #set the x and y axes to the same scale
  plt.xticks(np.arange(0, len(M), 1)) # remove the tick marks by setting to an empty list
  plt.yticks(np.arange(0, len(M[0]), 1)) # remove the tick marks by setting to an empty list
  plt.grid()
  plt.axes().invert_yaxis() #invert the y-axis so the first row of data is at the top

  plt.show()
# =============================================================

def main():
  np.random.seed(2)

  M = np.full((15,15), NORMAL)  # MAP
  shape = M.shape
  M = M.flatten()
  treasureIndecies = np.random.choice(M.size, size=20)
  M[treasureIndecies] = TREASURE
  M = M.reshape(shape)

  M[1][0] = TREASURE; M[2][1] = TREASURE; M[3][2] = TREASURE; M[4][3] = TREASURE

  M[5,1] = OBSTICLE; M[6,1] = OBSTICLE; M[7,1] = OBSTICLE; M[8,1] = OBSTICLE
  M[11,1] = OBSTICLE; M[12,1] = OBSTICLE; M[13,1] = OBSTICLE; M[14,1] = OBSTICLE
  M[11,2] = OBSTICLE; M[11,3] = OBSTICLE; M[11,4] = OBSTICLE; M[11,5] = OBSTICLE

  M[10,4] = OPPONENT; M[4,3] = OPPONENT; M[5,10] = OPPONENT; M[14,4] = OPPONENT

  M[0,0] = NORMAL
  M[14,14] = TREASURE


  

 
  
  print("Analyzing maze with RL Q-learning")
  start = 0
  max_steps = 10000

  ns = M.size # number of states
  goal = ns-1
  Q = np.zeros(shape=[ns,4], dtype=np.float32)  # Quality
    
  gamma = .5
  lrn_rate = .1
  max_epochs = 1000
  
  train(M, Q, gamma, lrn_rate, goal, ns, max_epochs, 0, max_steps)


  print("Step", np.argmax(Q[0]) )
  walk(start, goal, Q, M, max_steps)

if __name__ == "__main__":
  main()