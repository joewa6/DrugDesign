#!/bin/bash

#lepro 1jy1.pdb   ### OUTPUTS "pro.pdb" &  "dock.in" = ligand_list = ligands (name of mol2 files)

ledock dock.in
ledock -spli lig.dok
