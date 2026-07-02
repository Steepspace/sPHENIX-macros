#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}

source /opt/sphenix/core/bin/sphenix_setup.sh -n new

f4a_macro=${1}
input=${2}
output=${3}
output_tree=${4}
nEvents=${5}
dbtag=${6}
submitDir=${7}

# extract runnumber from file name
file=$(basename "$input")
IFS='-' read -r p1 p2 p3 <<< "$file"
run=$(echo "$p2" | sed 's/^0*//') # Remove leading zeros using sed
if [[ -z "$run" || ! "$run" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse run number from input: $input" >&2
    exit 1
fi

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }
    cut -d ',' -f 1 "$input" > dst_calofit.list
    cut -d ',' -f 2 "$input" > dst_zdc.list
    cut -d ',' -f 3 "$input" > dst_sepd.list
    getinputfiles.pl --verbose --filelist dst_calofit.list
    getinputfiles.pl --verbose --filelist dst_sepd.list

    # Create/clear a temporary file for the basenames
    > dst_zdc_local.list

    while IFS= read -r file; do
        # Skip empty lines if there are any
        [ -z "$file" ] && continue

        # Copy the file to the current directory
        cp -v "$file" .

        # Extract just the filename and save it to our local list
        basename "$file" >> dst_zdc_local.list
    done < dst_zdc.list

    # Overwrite the original list with the basename-only list
    mv dst_zdc_local.list dst_zdc.list

    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$run/hist" "$run/tree"

root -b -l -q "$f4a_macro($nEvents, \"dst_calofit.list\", \"dst_zdc.list\", \"dst_sepd.list\", \"$run/hist/$output\", \"$run/tree/$output_tree\", \"$dbtag\")"

echo "All Done and Transferring Files Back"

# Define maximum retries and a counter
max_retries=3
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv "$run" "$submitDir"; then
        success=1
        break
    else
        count=$((count + 1))
        echo "cp failed (likely GPFS lag). Retrying ($count/$max_retries) in 2 seconds..."
        sleep 2
    fi
done

if [ $success -eq 0 ]; then
    echo "Error: cp failed permanently after $max_retries attempts."
    exit 1
fi

echo "Finished"
