README – Supporting Information for
“Linkerability of Protein Ligands”

This archive contains the data and Python scripts used to perform
the linkerability analysis described in the manuscript.

CONTENTS


Python Scripts

LinkerabilityAnalysis.py
Main script used to perform the linkerability analysis of
protein–ligand complexes, including assessment of solvent
accessibility, local steric space, and solvent-directed geometry.

EntropyModel.py
Script used to calculate free energy penalties associated with
constraining flexible linkers within conical frustums based on
a geometry-based statistical mechanics model.



Data Files

Complexes.csv
Linkerability data for protein–ligand complexes analyzed from
the Protein Data Bank.

Atoms.csv
Linkerability data for individual modifiable positions on
protein-bound small molecules.

PPIs.csv
Linkerability data for ligands targeting protein–protein
interaction interfaces.

Kinases.csv
Linkerability data for ligands bound to protein kinases.

PARP2.csv
Linkerability data for ligands bound to PARP2.

MolecularGlues.csv
Linkerability data for ligands classified as molecular glues.

EnergyMap.csv
Precomputed entropic and free energy penalties for linkers
constrained within conical frustums across a range of cone
angles and heights.

REQUIREMENTS

Python libraries:
- numpy
- pandas
- rdkit
- biopython
- freesasa
- requests
- tqdm

DATA SOURCEs
Protein structures are from the Protein Data Bank (PDB).  
PDB IDs used are listed in Complexes.csv.


USAGE

Ensure all required Python packages and dependencies are installed.
Place all files in the same directory or update file paths
accordingly in the scripts.
Run LinkerabilityAnalysis.py to reproduce the main analysis.
EntropyModel.py can be run independently to compute free energy
penalties for linker confinement.

NOTES

The provided data files correspond to the datasets analyzed in
the manuscript and can be used to reproduce reported results.
File names and formats are consistent with those referenced in
the manuscript.
The computational models are simplified and intended to capture
general trends rather than provide quantitative predictions.

For questions, please contact Raphael.franzini@utah.edu