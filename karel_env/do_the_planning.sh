#!/bin/bash
#/Users/gavran/planners/fast-downward/fast-downward.py $1 $2 --search "astar(lmcut())"
timeout 30 $FASTDOWNWARD $1 $2 --search "astar(lmcut())"
#/Users/gavran/planners/fast-downward/fast-downward.py $1 $2 --search "lazy_greedy([ff()], preferred=[ff()])"