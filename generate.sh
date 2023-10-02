set -e
rm -rf comp || true
mkdir comp
g++ my_functions.cpp -g -std=c++17 -I ./Halide/include -L ./Halide/lib -lHalide -lpthread -ldl -o generate_my_functions
LD_LIBRARY_PATH=./Halide/lib ./generate_my_functions
rm -rf generate_my_functions