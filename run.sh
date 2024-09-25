#!/bin/bash

#PDB ID 1G6R (chain A )

neighbors=25

pdbname='5NRM'

chain='A'

PDB_path='example/PDB/'

Type='protein' 

testdata='5NRM_A.txt'

echo  Begin to extract features!

cd softwares

python ../codes/dssp.py $PDB_path $Type

cd ..

python ./codes/get_dssp.py $testdata $Type

python ./codes/prostT5.py $Type $testdata
    
python ./codes/consider_neighbor_nodes_edges.py $PDB_path 4$Type $testdata $neighbors

echo The calculation has been completed !

echo Begin to predicte binding sites !

python predicted.py $pdbname $chain $Type

rm -r example/features


