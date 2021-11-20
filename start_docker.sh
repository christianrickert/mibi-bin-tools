#!/usr/bin/env bash

# check for template developer flag
JUPYTER_DIR='scripts'
update=0
mount=''
while test $# -gt 0
do
  case "$1" in
    -u|--update)
      update=1
      shift
      ;;
    -m|--mount)
      mount="$2"
      shift
      shift
      ;;
    *)
      echo "$1 is not an accepted option..."
      echo "-u, --update                      : Update default scripts"
      echo "-m, --mount                       : Mount external drives to /data"
      exit
      ;;
  esac
done

# if requirements.txt has been changed in the last day, automatically rebuild Docker first
if [[ $(find . -mmin -1440 -type f -print | grep requirements.txt | wc -l) -eq 1 ]]
  then
    echo "New requirements.txt file detected, rebuilding Docker"
    docker build -t mibi-bin-tools .
fi

if [ ! -f "$PWD/$JUPYTER_DIR" ]
  then
    update=1
fi
if [ $update -ne 0 ]
  then
    bash update_notebooks.sh -u
  else
    bash update_notebooks.sh
fi

# find lowest open port available
PORT=8888

until [[ $(docker container ls | grep 0.0.0.0:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=$PORT+1))
done

if [ ! -z "$mount" ]
  then
    docker run -it \
      -p $PORT:$PORT \
      -v "$PWD/$JUPYTER_DIR:/$JUPYTER_DIR" \
      -v "$mount:/data/" \
      mibi-bin-tools:latest
  else
    echo "must provide a -m or --mount argument..."
    exit
fi
