#!/bin/bash

hipcc -o hip_rebar_bench hip_rebar_bench.cu -lpthread -lnuma
