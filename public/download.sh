#!/bin/bash

# data
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000364/data/data.tar.gz
tar -zxvf data.tar.gz

# code
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000364/data/code.tar.gz
tar -zxvf code.tar.gz
