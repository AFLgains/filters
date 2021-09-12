#!/usr/bin/env bash

PATH=$PATH:/c/anaconda3/envs/filters/
python -m filters -m live --no-email
python -m filters -m live --email -l ric.porteous1989@gmail.com

if [ $? -eq 0 ]; then
  echo 'All good!'
else
  echo 'An error has occured'
fi