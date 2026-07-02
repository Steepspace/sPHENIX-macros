#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}

source /opt/sphenix/core/bin/sphenix_setup.sh -n new

f4a_macro=${1}
input=${2}
QAhist=${3}
QVecCalibHist=${4}
pass=${5}
charge_threshold=${6}
noise_threshold=${7}
dst_tag=${8}
submitDir=${9}

# extract runnumber from file name
run=$(echo "$input" | grep -oP 'output/\K\d+(?=/tree)')
if [[ -z "$run" ]]; then
    echo "Failed to parse run number from input: $input" >&2
    exit 1
fi

QAhist_file=$(basename "$QAhist")
QVecCalibHist_file=$(basename "$QVecCalibHist")

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    cp -rv "$input" input
    readlink -f input/* > input.list

    cp -v "$QAhist" .
    test -e "$QVecCalibHist" && cp -v "$QVecCalibHist" .
    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "output/hist"

if [ "$pass" -eq 2 ]; then
    mkdir -p "output/CDB/$run"
fi

root -b -l -q "$f4a_macro(\"input.list\", \"$QAhist_file\", \"$QVecCalibHist_file\", $pass, 0, \"output/hist/QVecCalib-$run.root\", \"$dst_tag\", $charge_threshold, $noise_threshold, \"output/CDB/$run\")"

echo "All Done and Transferring Files Back"

# Define maximum retries and a counter
max_retries=3
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv output/* "$submitDir"; then
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
