#!/bin/bash
#BSUB -J cachejob         # Job name
#BSUB -q hpc           # Queue name
#BSUB -R "rusage[mem=20GB]" #ressources
#BSUB -o output.txt    # Output file
#BSUB -e error.txt     # Error file
#BSUB -W 00:10          # Walltime limit (hh:mm)
#BSUB -n 1
BSUB -N # get an email with all information, sometimes it contains more information

source ../venv3.9/bin/activate
python3 text_classification_vary.py