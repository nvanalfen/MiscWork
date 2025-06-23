redshifts=(-1 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
array_length=${#redshifts[@]}

for ((i=0; i<array_length-1; i++)); do
    min_z=${redshifts[i]}
    max_z=${redshifts[i+1]}
    if (( $(bc <<< "$max_z <= 1.0") )); then
        echo "Block 0"
        z_block=0
    elif (( $(bc <<< "$max_z <= 2.0") )); then
        echo "Block 1"
        z_block=1
    else
        echo "Block 2"
        z_block=2
    fi

    echo "BEGINNING RUN"
    echo "Run $i Start: Min z=$min_z, Max z=$max_z, Z block=$z_block"
    srun -n 32 python3 tidal_realign.py $min_z $max_z $z_block 1
    echo "Run $i End: Min z=$min_z, Max z=$max_z, Z block=$z_block"
    echo "COMPLETE RUN"
    echo "-----------------------------------"
    echo "-----------------------------------"
    echo "-----------------------------------"
done