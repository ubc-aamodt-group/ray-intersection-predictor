#!/bin/sh

cd build_make

for scene in ../scenes/*; do
    ./CWBVH --pathtrace --cwbvh --config ../scenes/$scene
    ./CWBVH --pathtrace --config ../scenes/$scene
done
