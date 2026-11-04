# Bug Algorithms
The bugs algorithms are a family **complete** and **online** algorithms based on similar strategies

They are designed to work only in **2-dimensional configuration spaces** and manage the presence of obstacles.
To adopt one of these algorithms, the robot requires self localization (through map, GPS, ...) and proximity sensing. Moreover the coordinates of the start and the destination point must be known. 
Bug algorithms combine 2 motion strategies:
- motion-to-goal
- boundary-following

The 3 types of bug algorithms are: 
### Bug 1
Until the robot reaches the destination point, it execute the following instructions:
- motion-to-goal until an obstacle is detected (hit point)
- the robot performs a complete circumnavigation of the obstacle (boundary-following) to find the point closest to the goal (leave point)
- the robot returns to the leave point along the shortest path and back to motion-to-goal.

### Bug 2
Until the robot reaches the destination point, it execute the following instructions:
- motion-to-goal until an obstacle is detected (hit point)
- the obstaacle is circumnavigated (boundary-following) until the r-line is encountered in a point closer to the goal than the previous hit point
- from the leave point the robot back to motion-to-goal.

_The r-line is a straight line connecting the staat point and the destination point._

### Tangent Bug
- Motion-to-goal strategy is used when the robot can move along the direction that minimizes a cost function (commonly based on distances), otherwise boundary-following is adopted.
- Note that differently from bug1 and bug 2: in motion-to-goal the robot can run along the obstacle boarder and in boundary-following the robot may travel without considering the obstacle border
- The behavior depends on the sensing range of the adopted range sensor (the ideal situation is an infinite range)

## Test 1 - Blob Obstacles
![blobs](bug_blobs.gif)

## Test 2 - Polygon Obstacles
![polygons](bug_polygons.gif)

## Test 3 - Spiral Obstacles
![spiral](bug_spiral.gif)

## Test 4 - Trap Scenario
![trap](bug_trap.gif)
