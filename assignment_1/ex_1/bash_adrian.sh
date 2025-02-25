#!/bin/bash
#BSUB -J ADLCV5         # Job name
#BSUB -q hpc           # Queue name
#BSUB -R "rusage[mem=40GB]" #ressources
#BSUB -o output4epoch10.txt    # Output file
#BSUB -e erroroutput4epoch10.txt     # Error file
#BSUB -W 05:10          # Walltime limit (hh:mm)
#BSUB -n 1
#BSUB -N # get an email with all information, sometimes it contains more information

source envcv/bin/activate
python3 text_classification_vary.py