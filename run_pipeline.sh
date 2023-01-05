#!/usr/bin/env bash

# author = 'Marcel Baltruschat'
# copyright = 'Copyright © 2020-2023'
# license = 'MIT'
# version = '1.1.0'

usage="\nUsage: %s --train SDF1 [SDF2 SDF3 ...] [--test SDF1 [SDF2 SDF3 ...]] [--no-openeye]\n\n"

if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  printf "$usage" "$0"
  exit 0
fi

train=0
test=0
train_files=()
test_files=()

for sdf in "$@"; do
  case $sdf in
    "--train")
      train=1
      test=0
    ;;
    "--test")
      train=0
      test=1
    ;;
    "--no-openeye")
      no_openeye="--no-openeye"
    ;;
    *)
      if [ $train -eq 0 ] && [ $test -eq 0 ]; then
        printf "\nInvalid call!"
        printf "$usage" "$0"
        exit 1
      fi
      if [ $train -eq 1 ]; then
        train_files+=("$sdf")
      else
        test_files+=("$sdf")
      fi
    ;;
  esac
done

cleaned_train_files=()
for sdf in "${train_files[@]}"; do
  cm_filename="$(basename "$sdf" .sdf)_cleaned_mono.sdf"
  cleaned_train_files+=("$cm_filename")
  echo
  echo "Cleaning, filtering and extracting all monoprotic structures ($(basename "$sdf"))..."
  python scripts/gen_clean_mono_dataset.py "$sdf" "$cm_filename" -kp pKa $no_openeye || exit 2
done

cleaned_test_files=()
for sdf in "${test_files[@]}"; do
  cm_filename="$(basename "$sdf" .sdf)_cleaned_mono.sdf"
  cleaned_test_files+=("$cm_filename")
  echo
  echo "Cleaning, filtering and extracting all monoprotic structures ($(basename "$sdf"))..."
  python scripts/gen_clean_mono_dataset.py "$sdf" "$cm_filename" -kp pKa $no_openeye || exit 2
done

echo
echo "Combining training datasets..."
python scripts/combine_datasets.py "${cleaned_train_files[@]}" || exit 3

echo
echo "Combining duplicates for training..."
python scripts/combine_mono_duplicates.py combined_training_datasets.sdf || exit 4

fin_test_files=()
for sdf in "${cleaned_test_files[@]}"; do
  bfn=$(basename "$sdf")
  echo
  echo "Combining duplicates for testset ($bfn)..."
  python scripts/combine_mono_duplicates.py "$bfn" || exit 5
  fin_test_files+=("$(basename "$bfn" .sdf)_unique.sdf")
done

if [ ${#fin_test_files[@]} -ne 0 ]; then
  echo
  echo "Remove training data from test files..."
  python scripts/remove_traindata_from_testdata.py combined_training_datasets_unique.sdf "${fin_test_files[@]}"
fi
