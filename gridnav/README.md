# Grid-Based Algorithms

The idea behind these algorithms is to map the enviroment. A map is a data structure that represents the enviroment where the robot moves. Mapping is the incremental process that builds a map using the information gathered from sensors. These algorithms are based on **occupancy grids** mapping:
- Grids are made by adjacent cells
- For each cell a flag indicates if an obstacles occupies the cell or not.
- The cell shape and size is chosen to match the enviroment and depends on the problem to solve.
  
The **resolution** (i.e. size of the cells) affects the success of the trajectory planning: an higher resolution increases the chance to find a path but it requires more memory space and more computing time.

Consider the case of square cells, the adjacency of two cells can be of two types:
- **4 point connectivity**, from the current cell the robot can only moves following the horizontal and vertical direction directions
- **8 point connectivity**, the robot can also moves following the diagonal directions

## Wavefront Algorithm
1. mark the cell containing the goal point with _i = 1_
2. mark with _i + 1_ every adjacent cell to the one marked with _1_
3. repeat _step 2_ until the cell containing the start point is marked or all the cell of the enviroment have been marked
4. use gradient descent to go from the starting cell to the goal cell (the path is not unique!)

The implementation of the wavefront algorithm requires the construction of a tree such that each node contains the coordinates and the distance information. The root corresponds to the destination node. The assignment of labels to the cells typically follows the breadth-first visit logic

Wavefront algorithm characteristics:
- complete: if a path exists (at a given resolution), it is found
- low efficiency: the breadth-first search is inefficient, a large amount of non necessary cells can be visited to assign the distance numbers
- optimal: it finds the shortest path

![wf](grid_wf_slide.gif)

## A* Algorithm
![astar](grid_astar_slide.gif)

## Rooms Test
![rooms](grid_rooms.gif)
