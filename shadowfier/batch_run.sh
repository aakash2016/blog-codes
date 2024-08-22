#!/bin/bash
file_path=$1

for f in $file_path/*
do
  if [ "${f: -4}" == ".jpg" ] || [ "${f: -4}" == ".png" ];
  then
    echo "$f"
    python -m main --path $f --show n --remove_bg n --type NORMAL
  fi
done
