#!/bin/bash

hipcc -o hip_rebar_bench hip_rebar_bench.cu -lpthread -lnuma

sleep 1

./hip_rebar_bench 100 0 1
