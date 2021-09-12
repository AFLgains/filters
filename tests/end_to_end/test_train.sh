#!/usr/bin/env bash

PASSWORD=jeanie8888
API_KEY=6e89c7206509df377432f33c9359bd07e11cc556b3c0976e4107336a648f4460
PATH=$PATH:/c/anaconda3/envs/filters/
python -m filters -m train --no-email
python -m filters -m train --email


if [ $? -eq 0 ]; then
  echo 'All good!'
else
  echo 'An error has occured'
fi