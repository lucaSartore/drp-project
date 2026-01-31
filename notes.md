#Assumtions

## robots movement:
1) Omnidirectional robot
2) unicycle

## runner movement
1) bouncing robot
2) actively evading the chasers

## Field of vision
1) 360 degrees
2) cone (after having made the motion an unicycle)

## runner probability mapping
1) use gaussian distribution
2) assume mesh (full) communication between the robots

## How robot choose movement
1) cooperative lioyd

## other ideas
1) add fake runners to false positive
2) evaluate different metrics (better more chasers with cheap sensors, or few chasers with good sensors)

# sources

https://www.sciencedirect.com/science/chapter/edited-volume/abs/pii/B9780128136775000067


# Distributed particle filter

## notations

x       = state
z       = measures
f(x|z)  = probability of state given measures
f(x|x)  = state transition probability (with xn & xn-1)


## Indexes

n = iteration (timestamp)
k = num agents
r = size of the set of approximation functions
i = iteration of the convergence procedure

