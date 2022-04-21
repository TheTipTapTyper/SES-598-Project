# SES-598-Project
Semester project for SES 598: Autonomous Exploration Systems. 
The goal is program a drone to autonomously explore a parking lot in order to find a vehicle.

In this project I implement the unsupervised learning method for terrain classification
described in this paper http://www.cim.mcgill.edu/~mrl/pubs/philg/crv2009.pdf

I use this method to train a classifier to distinguish between a parking lot and the
surrounding desert. I employ this classifier to allow a simulated drone to stay above
a parking lot while randomly exploring it.

In order to identify when the drone is above the target red vehicle, I created a simple
function using opencv which determines the portion of pixels in a image are red and then
thresholds then at a given value.

A demo video of the system can be viewed here: https://www.youtube.com/watch?v=8WJrmaRkGzg
