## Load Modules
If running this program on Clemson's Palmetto 2 cluster, run

    module load cuda gcc
    
## Compile & Run
Compile the program with
    
    nvcc -allow-unsupported-compiler -o bigDot2 bigDot2.cu

Run the program with

    ./bigDot2
