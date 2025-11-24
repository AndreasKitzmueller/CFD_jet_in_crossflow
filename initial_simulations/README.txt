folder elbow downloaded from the following repository:
https://github.com/OpenFOAM/OpenFOAM-dev/tree/master/tutorials/legacy/incompressible/icoFoam/elbow

activate the folder within the folder (if the path does not contain blanks or brackets):
source /opt/openfoam-dev/etc/bashrc

Remark: if the path contains blanks and brackets, temporarily copy the folder to the desktop

to create the polyMesh-folder within the constant-folder from elbow.msh:
fluentMeshToFoam elbow.msh

start the simulation (solver icoFoam) with the following command:
icoFoam

open the simulation in ParaView with the following command:
paraFoam

inside ParaView, click activate, and go through the p and u meshes
